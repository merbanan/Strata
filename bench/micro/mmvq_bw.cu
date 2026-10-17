// Build (server): nvcc -std=c++20 -ccbin g++-12 -arch=sm_70 -Iinclude bench/micro/mmvq_bw.cu build/libstrata_kernels.a build/libstrata_kernels_cpu.a -lcuda
// Bandwidth of the decode GEMVs on Flash-Next's dense shapes (one column): native_mmvq per type, bf16 for the hc
// projections. Weights filled with 0x11 bytes (finite scales); time = 200 launches, events.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/bf16_gemv.hpp"
using namespace strata::kernels;
int main() {
    struct T { int type; const char* name; } types[] = {{11, "Q3_K"}, {12, "Q4_K"}, {23, "IQ4_XS"}, {13, "Q5_K"}, {14, "Q6_K"}, {42, "Q2_0"}, {8, "Q8_0"}};
    struct S { int n_in, n_out; const char* what; } shapes[] = {{2560, 10240, "attn_qkv"}, {2560, 6144, "attn_gate"}, {6144, 2560, "ssm_out"}, {2560, 640, "shexp gate/up"}, {10240, 320, "hc down (as q8)"}, {320, 10240, "hc up (as q8)"}};
    void *w, *xq; float *y, *xf; uint16_t* wb;
    cudaMalloc(&w, 64 << 20); cudaMalloc(&xq, 1 << 20); cudaMalloc(&y, 1 << 20); cudaMalloc(&xf, 1 << 20); cudaMalloc(&wb, 64 << 20);
    cudaMemset(w, 0x11, 64 << 20); cudaMemset(wb, 0x11, 64 << 20); cudaMemset(xf, 0, 1 << 20);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaStream_t st; cudaStreamCreate(&st);
    for (const S& s : shapes)
        for (const T& t : types) {
            if (!native_mmvq_supported(t.type) || s.n_in % 256 && t.type != 8 && t.type != 42) continue;
            size_t bytes = 0;
            try { bytes = native_mmvq_weight_bytes(t.type, s.n_in, s.n_out); } catch (...) { continue; }
            native_quantize_q8_1(xf, xq, s.n_in, 1, st);
            for (int i = 0; i < 20; ++i) native_mmvq(t.type, w, xq, y, s.n_in, s.n_out, 1, st);
            cudaEventRecord(a, st);
            for (int i = 0; i < 200; ++i) native_mmvq(t.type, w, xq, y, s.n_in, s.n_out, 1, st);
            cudaEventRecord(b, st); cudaEventSynchronize(b);
            float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= 200;
            std::printf("%-16s %-7s %6.2f MB %7.1f us %6.0f GB/s\n", s.what, t.name, bytes / 1e6, ms * 1e3, bytes / (ms * 1e6));
        }
    for (S s : {S{10240, 320, "hc down bf16"}, S{320, 10240, "hc up bf16"}}) {
        const size_t bytes = (size_t) s.n_in * s.n_out * 2;
        for (int i = 0; i < 20; ++i) bf16_gemv_fp32_mmvf(xf, wb, y, s.n_in, s.n_out, st);
        cudaEventRecord(a, st);
        for (int i = 0; i < 200; ++i) bf16_gemv_fp32_mmvf(xf, wb, y, s.n_in, s.n_out, st);
        cudaEventRecord(b, st); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= 200;
        std::printf("%-16s %-7s %6.2f MB %7.1f us %6.0f GB/s\n", s.what, "BF16", bytes / 1e6, ms * 1e3, bytes / (ms * 1e6));
    }
    std::printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
