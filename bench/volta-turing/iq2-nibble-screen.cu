// RTX 2060 SUPER feasibility screen for lossless IQ2_XS signed nibbles.
// Same DP4A chain, integer scale arithmetic and FP32 reduction for both arms.
// Real gate/up rows from thirty saved experts; synthetic Q8 activations.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr int kRowsPerExpert = 1280, kWidth = 2560, kQ8Blocks = kWidth / 32;
constexpr int kOriginalBlock = 74, kNibbleBlock = 144;
constexpr int kOriginalRow = kWidth / 256 * kOriginalBlock;
constexpr int kNibbleRow = kWidth / 256 * kNibbleBlock;

void ck(cudaError_t e, const char* call) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", call, cudaGetErrorString(e)); std::exit(2); }
}
std::vector<uint8_t> read(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

template<bool NIBBLE>
__global__ void __launch_bounds__(128) row_dot(const uint8_t* __restrict__ weights,
                                                const block_q8_1* __restrict__ xq,
                                                float* __restrict__ output, int experts) {
    constexpr int row_bytes = NIBBLE ? kNibbleRow : kOriginalRow;
    constexpr int block_bytes = NIBBLE ? kNibbleBlock : kOriginalBlock;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + warp, rows = experts * kRowsPerExpert;
    if (row >= rows) return;
    const uint8_t* wr = weights + (size_t) row * row_bytes;
    const block_q8_1* q = xq + (size_t) blockIdx.y * kQ8Blocks;
    float sum = 0.0f;
    for (int k = lane; k < 80; k += 32) {
        const int kbx = k >> 3, iqs = 2 * (k & 7);
        const uint8_t* block = wr + kbx * block_bytes;
        const block_q8_1* a = q + kbx * 8 + iqs / 2;
        const uint8_t sc = block[NIBBLE ? 2 + iqs / 2 : 66 + iqs / 2];
        const int ls0 = sc & 15, ls1 = sc >> 4;
        int sumi0 = 0, sumi1 = 0;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            uint32_t g0, g1;
            if constexpr (NIBBLE) {
                const uint32_t packed = *reinterpret_cast<const uint32_t*>(block + 16 + 16 * (iqs / 2) + 4 * (l0 / 2));
                // Byte choices 0..5 represent {-43,-25,-8,8,25,43}. The low
                // and high 16 bits already contain four selectors each.
                g0 = __byte_perm(0x08f8e7d5u, 0x00002b19u, packed & 0xffffu);
                g1 = __byte_perm(0x08f8e7d5u, 0x00002b19u, packed >> 16);
            } else {
                const uint16_t code = *reinterpret_cast<const uint16_t*>(block + 2 + 2 * (2 * iqs + l0 / 2));
                const uint64_t mag = iq2xs_grid[code & 511];
                const uint32_t signs = (uint32_t) ksigns_iq2xs[code >> 9] * 0x01010101u;
                const uint32_t mask0 = __vcmpne4(signs & 0x08040201u, 0);
                const uint32_t mask1 = __vcmpne4(signs & 0x80402010u, 0);
                g0 = __vsub4((uint32_t) mag ^ mask0, mask0);
                g1 = __vsub4((uint32_t) (mag >> 32) ^ mask1, mask1);
            }
            const int a0 = reinterpret_cast<const int*>(a->qs)[l0];
            const int a1 = reinterpret_cast<const int*>(a->qs)[l0 + 1];
            if (l0 < 4) {
                sumi0 = __dp4a((int) g0, a0, sumi0);
                sumi0 = __dp4a((int) g1, a1, sumi0);
            } else {
                sumi1 = __dp4a((int) g0, a0, sumi1);
                sumi1 = __dp4a((int) g1, a1, sumi1);
            }
        }
        const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
        const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(block))) * __low2float(a->ds);
        sum += d * sumi;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) output[(size_t) blockIdx.y * rows + row] = sum;
}

template<bool NIBBLE>
void launch(const uint8_t* weights, const block_q8_1* xq, float* output, int experts, int tokens) {
    row_dot<NIBBLE><<<dim3((unsigned) ((experts * kRowsPerExpert + 3) / 4), (unsigned) tokens), 128>>>(
        weights, xq, output, experts);
    ck(cudaGetLastError(), "row_dot launch");
}
uint64_t hash(const std::vector<float>& v) {
    uint64_t h = 14695981039346656037ull;
    for (float x : v) {
        uint32_t bits = std::bit_cast<uint32_t>(x);
        for (int s = 0; s < 32; s += 8) { h ^= (bits >> s) & 255u; h *= 1099511628211ull; }
    }
    return h;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3 && !(argc == 4 && std::string(argv[3]) == "--small")) {
        std::fprintf(stderr, "usage: %s ORIGINAL-GU-ALL CONVERTED-GU-ALL [--small]\n", argv[0]); return 2;
    }
    const bool small = argc == 4;
    const auto original = read(argv[1]), nibble = read(argv[2]);
    if (original.size() % (kRowsPerExpert * kOriginalRow) ||
        nibble.size() != original.size() / (kRowsPerExpert * kOriginalRow) * kRowsPerExpert * kNibbleRow) {
        std::fprintf(stderr, "weight sizes do not match complete experts\n"); return 2;
    }
    const int total_experts = (int) (original.size() / (kRowsPerExpert * kOriginalRow));
    cudaDeviceProp device{}; ck(cudaGetDeviceProperties(&device, 0), "device");
    std::printf("DEVICE,%s,%d,%d,experts=%d,original_bytes=%zu,nibble_bytes=%zu\n",
                device.name, device.major, device.minor, total_experts, original.size(), nibble.size());
    uint8_t *dw0 = nullptr, *dw1 = nullptr;
    block_q8_1* dq = nullptr;
    float *do0 = nullptr, *do1 = nullptr;
    ck(cudaMalloc(&dw0, original.size()), "malloc original");
    ck(cudaMalloc(&dw1, nibble.size()), "malloc nibble");
    ck(cudaMemcpy(dw0, original.data(), original.size(), cudaMemcpyHostToDevice), "upload original");
    ck(cudaMemcpy(dw1, nibble.data(), nibble.size(), cudaMemcpyHostToDevice), "upload nibble");
    std::mt19937 rng(20261023);
    std::uniform_int_distribution<int> code(-127, 127);
    const int max_tokens = 8;
    std::vector<block_q8_1> q((size_t) max_tokens * kQ8Blocks);
    for (auto& block : q) {
        const __half h = __float2half_rn(0.004f + (rng() % 1000) * 0.00002f);
        block.ds = __halves2half2(h, h);
        for (auto& x : block.qs) x = (int8_t) code(rng);
    }
    ck(cudaMalloc(&dq, q.size() * sizeof(block_q8_1)), "malloc q8");
    ck(cudaMemcpy(dq, q.data(), q.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice), "upload q8");
    const size_t max_outputs = (size_t) total_experts * kRowsPerExpert * max_tokens;
    ck(cudaMalloc(&do0, max_outputs * sizeof(float)), "malloc output original");
    ck(cudaMalloc(&do1, max_outputs * sizeof(float)), "malloc output nibble");
    cudaEvent_t begin, end;
    ck(cudaEventCreate(&begin), "event begin"); ck(cudaEventCreate(&end), "event end");
    for (int experts : (small ? std::vector<int>{1} : std::vector<int>{1, 3, total_experts}))
      for (int tokens : (small ? std::vector<int>{1, 8} : std::vector<int>{1, 4, 8})) {
        const size_t count = (size_t) experts * kRowsPerExpert * tokens;
        launch<false>(dw0, dq, do0, experts, tokens);
        launch<true>(dw1, dq, do1, experts, tokens);
        ck(cudaDeviceSynchronize(), "parity synchronize");
        std::vector<float> out0(count), out1(count);
        ck(cudaMemcpy(out0.data(), do0, count * sizeof(float), cudaMemcpyDeviceToHost), "download original");
        ck(cudaMemcpy(out1.data(), do1, count * sizeof(float), cudaMemcpyDeviceToHost), "download nibble");
        size_t bad = 0;
        for (size_t i = 0; i < count; ++i) bad += std::bit_cast<uint32_t>(out0[i]) != std::bit_cast<uint32_t>(out1[i]);
        std::printf("PARITY,%d,%d,%zu,%zu,%016llx,%016llx\n", experts, tokens, count, bad,
                    (unsigned long long) hash(out0), (unsigned long long) hash(out1));
        if (bad) return 1;
        for (int pair = 0; pair < 2; ++pair) {
            for (int slot = 0; slot < 4; ++slot) {
                const bool is_nibble = pair ? slot < 2 : slot >= 2;
                for (int warm = 0; warm < 3; ++warm)
                    if (is_nibble) launch<true>(dw1, dq, do1, experts, tokens);
                    else launch<false>(dw0, dq, do0, experts, tokens);
                ck(cudaEventRecord(begin), "event start");
                if (is_nibble) launch<true>(dw1, dq, do1, experts, tokens);
                else launch<false>(dw0, dq, do0, experts, tokens);
                ck(cudaEventRecord(end), "event stop");
                ck(cudaEventSynchronize(end), "event wait");
                float ms = 0; ck(cudaEventElapsedTime(&ms, begin, end), "elapsed");
                std::printf("TIME,%d,%d,%d,%d,%s,%.6f\n", experts, tokens, pair, slot,
                            is_nibble ? "nibble" : "original", ms);
            }
        }
    }
    ck(cudaEventDestroy(begin), "event destroy"); ck(cudaEventDestroy(end), "event destroy");
    ck(cudaFree(dw0), "free original"); ck(cudaFree(dw1), "free nibble");
    ck(cudaFree(dq), "free q8"); ck(cudaFree(do0), "free output original"); ck(cudaFree(do1), "free output nibble");
    return 0;
}
