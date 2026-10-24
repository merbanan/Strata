// Standalone verifier-sized CUDA graph update probe. No model memory required.
// nvcc -O2 -std=c++17 -arch=sm_75 graph-same-size-update.cu -o graph-same-size-update
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
    std::exit(2); } } while (0)

__global__ void step(unsigned* output, unsigned delta) {
    output[threadIdx.x] += delta + threadIdx.x;
}
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
static cudaGraph_t capture(cudaStream_t stream, unsigned* output, int nodes, unsigned base) {
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < nodes; ++i) step<<<1, 32, 0, stream>>>(output, base + i);
    CHECK(cudaGetLastError());
    cudaGraph_t graph;
    CHECK(cudaStreamEndCapture(stream, &graph));
    size_t count = 0;
    CHECK(cudaGraphGetNodes(graph, nullptr, &count));
    if (count != static_cast<size_t>(nodes)) std::exit(3);
    return graph;
}
static void replay(cudaStream_t stream, cudaGraphExec_t exec, unsigned* output,
                   int nodes, unsigned base, unsigned seed) {
    unsigned input[32], result[32];
    for (unsigned lane = 0; lane < 32; ++lane) input[lane] = seed + lane;
    CHECK(cudaMemcpyAsync(output, input, sizeof(input), cudaMemcpyHostToDevice, stream));
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaMemcpyAsync(result, output, sizeof(result), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
    for (unsigned lane = 0; lane < 32; ++lane) {
        unsigned expected = input[lane] + static_cast<unsigned>(nodes) * (base + lane)
                          + static_cast<unsigned>(nodes) * (nodes - 1) / 2;
        if (result[lane] != expected) {
            std::fprintf(stderr, "mismatch lane=%u got=%u expected=%u\n", lane, result[lane], expected);
            std::exit(4);
        }
    }
}
int main() {
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("gpu=%s\n", prop.name);
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    unsigned* output;
    CHECK(cudaMalloc(&output, 32 * sizeof(unsigned)));
    constexpr int nodes = 4307;
    step<<<1, 32, 0, stream>>>(output, 0);
    CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t a = capture(stream, output, nodes, 17);
    cudaGraphExec_t exec;
    auto start = Clock::now();
    CHECK(cudaGraphInstantiate(&exec, a, 0));
    std::printf("instantiate_a_ms=%.3f\n", ms(start));
    replay(stream, exec, output, nodes, 17, 101);
    cudaGraph_t b = capture(stream, output, nodes, 53);
    cudaGraphExecUpdateResultInfo info{};
    start = Clock::now();
    cudaError_t update = cudaGraphExecUpdate(exec, b, &info);
    std::printf("same_size_update_ms=%.3f status=%s result=%d\n",
                ms(start), cudaGetErrorString(update), static_cast<int>(info.result));
    if (update != cudaSuccess || info.result != cudaGraphExecUpdateSuccess) return 5;
    replay(stream, exec, output, nodes, 53, 202);
    replay(stream, exec, output, nodes, 53, 303);

    cudaGraphExec_t fresh;
    start = Clock::now();
    CHECK(cudaGraphInstantiate(&fresh, b, 0));
    std::printf("fresh_b_instantiate_ms=%.3f\n", ms(start));
    replay(stream, fresh, output, nodes, 53, 404);

    cudaGraph_t c = capture(stream, output, nodes - 1, 89);
    start = Clock::now();
    update = cudaGraphExecUpdate(exec, c, &info);
    std::printf("cross_size_update_ms=%.3f status=%s result=%d\n",
                ms(start), cudaGetErrorString(update), static_cast<int>(info.result));
    if (update != cudaErrorGraphExecUpdateFailure) return 6;
    replay(stream, exec, output, nodes, 53, 505);

    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphExecDestroy(fresh));
    CHECK(cudaGraphDestroy(a));
    CHECK(cudaGraphDestroy(b));
    CHECK(cudaGraphDestroy(c));
    CHECK(cudaFree(output));
    CHECK(cudaStreamDestroy(stream));
    std::puts("PASS");
}
