// Standalone CUDA graph metadata retention experiment, not an engine benchmark.
// nvcc -O2 -std=c++17 -arch=sm_75 graph-retention-probe.cu -o graph-retention-probe
// Usage: graph-retention-probe CAP [HEADROOM_MIB]; CAP is 1..5.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
    std::exit(2); } } while (0)

struct Params { unsigned long long words[16]; };
__global__ void step(unsigned* output, Params params) {
    unsigned lane = threadIdx.x;
    output[lane] += static_cast<unsigned>(params.words[lane % 16]) ^ lane;
}

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}
static int live = 0;
static int creations = 0;
static void sample(const char* event, int phase, int shape, double duration = 0) {
    size_t available = 0, total = 0;
    CHECK(cudaMemGetInfo(&available, &total));
    unsigned long long used = 0, reserved = 0;
    CHECK(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrUsedMemCurrent, &used));
    CHECK(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrReservedMemCurrent, &reserved));
    std::printf("{\"event\":\"%s\",\"phase\":%d,\"shape\":%d,\"live\":%d,"
                "\"creations\":%d,\"free\":%zu,\"total\":%zu,\"pool_used\":%llu,"
                "\"pool_reserved\":%llu,\"duration_ms\":%.6f}\n",
                event, phase, shape, live, creations, available, total, used, reserved, duration);
    std::fflush(stdout);
}

struct Entry { cudaGraphExec_t exec = nullptr; int touched = -1; };
int main(int argc, char** argv) {
    int cap = argc > 1 ? std::atoi(argv[1]) : 0;
    int headroom = argc > 2 ? std::atoi(argv[2]) : 0;
    if (argc > 3 || cap < 1 || cap > 5 || headroom < 0) {
        std::fprintf(stderr, "usage: graph-retention-probe CAP(1..5) [HEADROOM_MIB]\n");
        return 1;
    }
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int runtime = 0, driver = 0;
    CHECK(cudaRuntimeGetVersion(&runtime)); CHECK(cudaDriverGetVersion(&driver));
    std::printf("{\"event\":\"config\",\"cap\":%d,\"headroom_mib\":%d,\"gpu\":\"%s\","
                "\"runtime\":%d,\"driver\":%d,\"parameter_bytes\":%zu}\n",
                cap, headroom, prop.name, runtime, driver, sizeof(Params));
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    unsigned* output = nullptr;
    CHECK(cudaMalloc(&output, 32 * sizeof(unsigned)));
    CHECK(cudaMemset(output, 0, 32 * sizeof(unsigned)));
    step<<<1,32,0,stream>>>(output, Params{});
    CHECK(cudaGetLastError()); CHECK(cudaStreamSynchronize(stream));
    sample("initialized", -1, -1);
    void* ballast = nullptr;
    if (headroom) {
        size_t available = 0, total = 0;
        CHECK(cudaMemGetInfo(&available, &total));
        size_t reserve = static_cast<size_t>(headroom) * 1024 * 1024;
        if (available <= reserve) return 3;
        size_t bytes = (available - reserve) / (2 * 1024 * 1024) * (2 * 1024 * 1024);
        CHECK(cudaMalloc(&ballast, bytes));
        CHECK(cudaMemsetAsync(ballast, 0xA5, bytes, stream));
        CHECK(cudaStreamSynchronize(stream));
        std::printf("{\"event\":\"ballast\",\"bytes\":%zu}\n", bytes);
    }
    sample("start", -1, -1);
    constexpr int nodes[5] = {4307, 2300, 4440, 4300, 4346};
    constexpr int sequence[] = {0,1,2,3,4,0,2,4,1,3,0,1,2,3,4,0,2,4,1,3};
    std::array<Entry, 5> entries{};
    std::array<std::array<unsigned, 32>, 5> deltas{};
    for (int shape = 0; shape < 5; ++shape)
        for (int n = 0; n < nodes[shape]; ++n)
            for (unsigned lane = 0; lane < 32; ++lane)
                deltas[shape][lane] += (static_cast<unsigned>((n+1)*17 + (lane%16)*13 + shape*101) ^ lane);
    auto destroy = [&](int shape, int phase) {
        CHECK(cudaStreamSynchronize(stream));
        sample("before_destroy", phase, shape);
        auto begin = Clock::now();
        CHECK(cudaGraphExecDestroy(entries[shape].exec));
        entries[shape].exec = nullptr; --live;
        sample("destroy", phase, shape, ms(begin));
        CHECK(cudaDeviceSynchronize());
        sample("destroy_synced", phase, shape);
    };
    int checked = 0, tick = 0;
    // Phase 1 starts in the same CUDA context after destroying every exec.
    // Phase 2 also trims the separate graph-allocation-node pool first.
    for (int phase = 0; phase < 3; ++phase) {
        for (int shape : sequence) {
            if (!entries[shape].exec) {
                if (live == cap) {
                    int victim = -1;
                    for (int i = 0; i < 5; ++i)
                        if (entries[i].exec && (victim == -1 || entries[i].touched < entries[victim].touched)) victim = i;
                    destroy(victim, phase);
                }
                sample("before_capture", phase, shape);
                CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                for (int n = 0; n < nodes[shape]; ++n) {
                    Params params{};
                    for (int k = 0; k < 16; ++k) params.words[k] = (n+1)*17 + k*13 + shape*101;
                    step<<<1,32,0,stream>>>(output, params);
                }
                CHECK(cudaGetLastError());
                cudaGraph_t graph;
                CHECK(cudaStreamEndCapture(stream, &graph));
                size_t count = 0;
                CHECK(cudaGraphGetNodes(graph, nullptr, &count));
                if (count != static_cast<size_t>(nodes[shape])) return 4;
                sample("before_instantiate", phase, shape);
                auto begin = Clock::now();
                CHECK(cudaGraphInstantiate(&entries[shape].exec, graph, 0));
                ++live; ++creations;
                sample("instantiate", phase, shape, ms(begin));
                CHECK(cudaGraphUpload(entries[shape].exec, stream));
                CHECK(cudaStreamSynchronize(stream));
                sample("upload_synced", phase, shape);
                CHECK(cudaGraphDestroy(graph));
            }
            entries[shape].touched = tick++;
            // Changed inputs and repeated launches verify retained AND recreated graphs.
            std::array<unsigned, 32> seed{}, actual{};
            for (unsigned lane = 0; lane < 32; ++lane) seed[lane] = 0xf0000000u + tick * 123 + lane;
            CHECK(cudaMemcpyAsync(output, seed.data(), sizeof(seed), cudaMemcpyHostToDevice, stream));
            for (int replay = 0; replay < 3; ++replay) CHECK(cudaGraphLaunch(entries[shape].exec, stream));
            CHECK(cudaMemcpyAsync(actual.data(), output, sizeof(actual), cudaMemcpyDeviceToHost, stream));
            CHECK(cudaStreamSynchronize(stream));
            for (unsigned lane = 0; lane < 32; ++lane) {
                unsigned expected = seed[lane] + 3u * deltas[shape][lane];
                if (actual[lane] != expected) {
                    std::fprintf(stderr, "mismatch phase=%d shape=%d lane=%u got=%u expected=%u\n",
                                 phase, shape, lane, actual[lane], expected);
                    return 5;
                }
            }
            ++checked;
            sample("verified", phase, shape);
        }
        for (int i = 0; i < 5; ++i) if (entries[i].exec) destroy(i, phase);
        sample("phase_destroyed", phase, -1);
        if (phase == 1) {
            CHECK(cudaDeviceGraphMemTrim(0)); CHECK(cudaDeviceSynchronize());
            sample("trim_synced", phase, -1);
        }
    }
    CHECK(cudaFree(ballast)); CHECK(cudaFree(output)); CHECK(cudaStreamDestroy(stream));
    sample("finished", 3, -1);
    std::printf("{\"event\":\"pass\",\"checked\":%d,\"launches\":%d,\"creations\":%d}\n",
                checked, checked*3, creations);
    return 0;
}
