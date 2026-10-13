// src/kernels/mmvq_tc_bench.cpp - mmvq_tc (STRATA_MMVQ_TC) against native_mmvq: relative error, and the time per call
// of the tensor-core kernel, the interleaved IL kernels (native_mmvq_il) and the plain multi-column kernels, T = 1..8.
//
//     build/mmvq_tc_bench [filter]
//
// Weights are random bytes with the K-quant d/dmin rewritten as normal fp16 values; the bench cycles through enough
// copies of the weights to miss the L2 cache, as a layer's call does.
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::kernels;
namespace {
struct Case { const char* name; int type, n_in, n_out, bb; };
const Case CASES[] = {
    {"Q4_K 2560x10240", 12, 2560, 10240, 144}, {"Q4_K 6144x2560", 12, 6144, 2560, 144},
    {"Q4_K 2560x6144", 12, 2560, 6144, 144},   {"Q4_K 2560x640", 12, 2560, 640, 144},
    {"Q5_K 2560x12288", 13, 2560, 12288, 176}, {"Q5_K 6144x2560", 13, 6144, 2560, 176},
    {"Q5_K 2560x10240", 13, 2560, 10240, 176}, {"Q5_K head 2560x248320", 13, 2560, 248320, 176},
    {"Q6_K 2560x10240", 14, 2560, 10240, 210}, {"Q6_K 6144x2560", 14, 6144, 2560, 210},
    {"Q6_K 2560x12288", 14, 2560, 12288, 210}, {"Q6_K 2560x6144", 14, 2560, 6144, 210},
};
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s: %s\n", #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)

template <class F> double time_us(F f, int reps, cudaStream_t s) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    f(0); cudaStreamSynchronize(s);
    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        cudaEventRecord(a, s);
        for (int i = 0; i < reps; ++i) f(i);
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        best = std::min(best, 1000.0 * ms / reps);
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best;
}
}  // namespace

// --real <type> <weights.bin> <n_in> <n_out> <x.bin> <T> <out prefix>: real GGUF rows, given fp32 activations (T x n_in);
// writes <prefix>_ref.bin (native_mmvq) and <prefix>_tc.bin (mmvq_tc), T x n_out fp32 each
static int real_mode(char** a) {
    const int type = std::atoi(a[0]), n_in = std::atoi(a[2]), n_out = std::atoi(a[3]), T = std::atoi(a[5]);
    auto slurp = [](const char* f) { std::vector<uint8_t> v; FILE* fp = std::fopen(f, "rb"); if (!fp) return v;
        std::fseek(fp, 0, SEEK_END); v.resize(std::ftell(fp)); std::fseek(fp, 0, SEEK_SET);
        if (std::fread(v.data(), 1, v.size(), fp) != v.size()) v.clear(); std::fclose(fp); return v; };
    auto hw = slurp(a[1]), hx = slurp(a[4]);
    if (hw.empty() || hx.size() < (size_t) n_in * T * 4) { std::printf("bad inputs\n"); return 2; }
    cudaStream_t s; cudaStreamCreate(&s);
    void *dw, *xq; float *dx, *yr, *yt;
    CK(cudaMalloc(&dw, hw.size())); CK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dx, (size_t) n_in * T * 4)); CK(cudaMemcpy(dx, hx.data(), (size_t) n_in * T * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&xq, native_q8_1_bytes(n_in, T)));
    CK(cudaMalloc(&yr, (size_t) n_out * T * 4)); CK(cudaMalloc(&yt, (size_t) n_out * T * 4));
    native_quantize_q8_1(dx, xq, n_in, T, s);
    native_mmvq(type, dw, xq, yr, n_in, n_out, T, s);
    if (!native_mmvq_tc(type, dw, xq, yt, n_in, n_out, T, s, true)) { std::printf("tc refused\n"); return 2; }
    CK(cudaStreamSynchronize(s));
    std::vector<float> o((size_t) n_out * T);
    for (int k = 0; k < 2; ++k) {
        CK(cudaMemcpy(o.data(), k ? yt : yr, o.size() * 4, cudaMemcpyDeviceToHost));
        const std::string f = std::string(a[6]) + (k ? "_tc.bin" : "_ref.bin");
        FILE* fp = std::fopen(f.c_str(), "wb"); std::fwrite(o.data(), 4, o.size(), fp); std::fclose(fp);
    }
    std::printf("real: wrote %s_ref.bin / _tc.bin\n", a[6]);
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 7 && std::string(argv[1]) == "--real") return real_mode(argv + 2);
    const char* only = argc > 1 ? argv[1] : nullptr;
    cudaStream_t s; cudaStreamCreate(&s);
    std::mt19937 rng(1234);
    int bad = 0;
    std::printf("%-24s %2s %9s %9s %9s %7s %7s %10s\n", "case", "T", "plain us", "il us", "tc us", "tc GB/s", "x best", "rel err");
    for (const Case& c : CASES) {
        if (only && !std::strstr(c.name, only)) continue;
        const size_t wbytes = (size_t) c.n_out * (c.n_in / 256) * c.bb;
        const int copies = (int) std::max<size_t>(1, std::min<size_t>(8, (64ull << 20) / wbytes + 1));
        std::vector<uint8_t> hw(wbytes);
        for (auto& v : hw) v = (uint8_t) rng();
        for (size_t blk = 0; blk < wbytes / c.bb; ++blk) {   // d (and dmin): normal fp16
            if (c.type == 14) {   // Q6_K: d at byte 208
                half d = __float2half(0.001f + 0.004f * (rng() % 1000) / 1000.0f);
                std::memcpy(&hw[blk * c.bb + 208], &d, 2);
                continue;
            }
            half2 dm = __floats2half2_rn(0.001f + 0.004f * (rng() % 1000) / 1000.0f, 0.0005f * (rng() % 1000) / 1000.0f);
            std::memcpy(&hw[blk * c.bb], &dm, 4);
        }
        std::vector<void*> dw(copies);
        for (auto& p : dw) { CK(cudaMalloc(&p, wbytes)); CK(cudaMemcpy(p, hw.data(), wbytes, cudaMemcpyHostToDevice)); }
        const int TMAX = 8;
        std::vector<float> hx((size_t) c.n_in * TMAX);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : hx) v = nd(rng);
        float* dx; void* xq; void* xil; float *yr, *yt, *yi;
        CK(cudaMalloc(&dx, hx.size() * 4)); CK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&xq, native_q8_1_bytes(c.n_in, TMAX)));
        CK(cudaMalloc(&xil, native_q8_1_bytes(c.n_in, TMAX) * 2));
        CK(cudaMalloc(&yr, (size_t) c.n_out * TMAX * 4)); CK(cudaMalloc(&yt, (size_t) c.n_out * TMAX * 4)); CK(cudaMalloc(&yi, (size_t) c.n_out * TMAX * 4));
        for (int T = 1; T <= TMAX; ++T) {
            native_quantize_q8_1(dx, xq, c.n_in, T, s);
            if (T >= 2 && T <= 4) native_q8_1_interleave(xq, xil, c.n_in, T, s);
            native_mmvq(c.type, dw[0], xq, yr, c.n_in, c.n_out, T, s);
            if (!native_mmvq_tc(c.type, dw[0], xq, yt, c.n_in, c.n_out, T, s, true)) { std::printf("tc refused\n"); return 2; }
            CK(cudaStreamSynchronize(s));
            std::vector<float> a((size_t) c.n_out * T), b(a.size());
            CK(cudaMemcpy(a.data(), yr, a.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), yt, b.size() * 4, cudaMemcpyDeviceToHost));
            double num = 0, den = 0; size_t nonfin = 0;
            for (size_t i = 0; i < a.size(); ++i) { if (!std::isfinite(b[i])) ++nonfin; num += (a[i] - b[i]) * (double) (a[i] - b[i]); den += (double) a[i] * a[i]; }
            const double rel = std::sqrt(num / std::max(den, 1e-30));
            if (rel > 1e-3 || nonfin) ++bad;
            const int reps = std::max(4, (int) (2e8 / wbytes));
            const double tp = time_us([&](int i) { native_mmvq(c.type, dw[i % copies], xq, yr, c.n_in, c.n_out, T, s); }, reps, s);
            double ti = -1;
            if (T <= 4 && native_mmvq_il_supported(c.type, T, c.n_out))
                ti = time_us([&](int i) { native_mmvq_il(c.type, dw[i % copies], xq, xil, yi, c.n_in, c.n_out, T, s); }, reps, s);
            const double tt = time_us([&](int i) { native_mmvq_tc(c.type, dw[i % copies], xq, yt, c.n_in, c.n_out, T, s, true); }, reps, s);
            const double best = ti > 0 ? std::min(tp, ti) : tp;
            std::printf("%-24s %2d %9.1f %9.1f %9.1f %7.0f %6.2fx %10.2e%s\n", c.name, T, tp, ti, tt, wbytes / tt / 1e3,
                        best / tt, rel, nonfin ? " NONFINITE" : "");
        }
        for (auto p : dw) cudaFree(p);
        cudaFree(dx); cudaFree(xq); cudaFree(xil); cudaFree(yr); cudaFree(yt); cudaFree(yi);
    }
    std::printf("%s\n", bad ? "mmvq_tc_bench: ACCURACY FAIL" : "mmvq_tc_bench: OK");
    return bad ? 1 : 0;
}
