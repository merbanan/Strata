// tools/hc_bench.cu - times fused_gr_read_multi (the hyper-connection read) for T tokens on random weights,
// cycling 4 weight sets (as 4 layers would) so nothing stays in L2. Reports effective BF16 weight bandwidth and a
// checksum of the outputs for comparing variants.  usage: hc_bench [T=2] [reps=200]
#include "strata/kernels/fused_gr.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::kernels;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;

static uint16_t bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x8000u) >> 16); }
template<typename T> T* up(const std::vector<T>& h) { T* d; CK(cudaMalloc(&d, h.size() * sizeof(T))); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }

int main(int argc, char** argv) {
    const int T = argc > 1 ? std::atoi(argv[1]) : 2;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 200;
    constexpr int SETS = 4;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto randv = [&](size_t n, float s) { std::vector<float> v(n); for (auto& x : v) x = nd(rng) * s; return v; };
    auto randb = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto& x : v) x = bf16(nd(rng) * s); return v; };
    const bool q8 = false;   // BF16 weights (upstream's STRATA_HC_Q8 reads Q8_0 bytes through its own kernels)
    struct Set { float* wn; uint16_t *wd, *wu, *wi; float *sd = nullptr, *su = nullptr; } sets[SETS];
    auto q8mat = [&](size_t n, float sc, float*& scales) {
        std::vector<int8_t> q(n);
        std::vector<float> d(n / 32);
        for (size_t b = 0; b < n / 32; ++b) {
            float v[32], amax = 0;
            for (int j = 0; j < 32; ++j) { v[j] = nd(rng) * sc; amax = std::max(amax, std::fabs(v[j])); }
            d[b] = amax / 127.f;
            for (int j = 0; j < 32; ++j) q[b * 32 + j] = (int8_t) std::nearbyint(v[j] / d[b]);
        }
        scales = up(d);
        return (uint16_t*) up(q);
    };
    for (auto& s : sets) {
        s.wn = up(randv(D, 1.f));
        if (q8) {
            s.wd = q8mat((size_t) LR * D, 0.02f, s.sd);
            s.wu = q8mat((size_t) D * LR, 0.05f, s.su);
        } else {
            s.wd = up(randb((size_t) LR * D, 0.02f));
            s.wu = up(randb((size_t) D * LR, 0.05f));
        }
        s.wi = up(randb((size_t) HC * D, 0.02f));
    }
    std::vector<FusedGrArgs> a(T);
    std::vector<float*> R(T), mixed(T), inj(T);
    float* bo = up(randv(N, 0.5f));
    float* injp = up(randv(HC, 1.f));
    for (int t = 0; t < T; ++t) {
        R[t] = up(randv(D, 1.f));
        CK(cudaMalloc(&mixed[t], N * 4));
        CK(cudaMalloc(&inj[t], HC * 4));
        float *lo, *rs;
        CK(cudaMalloc(&lo, LR * 4));
        CK(cudaMalloc(&rs, HC * 4));
        a[t].R = R[t]; a[t].R_out = R[t]; a[t].apply = false; a[t].bo_prev = bo; a[t].inj_prev = injp;
        a[t].lo = lo; a[t].rs = rs; a[t].inject_out = inj[t]; a[t].mixed = mixed[t];
    }
    float* xn;
    CK(cudaMalloc(&xn, (size_t) T * D * 4));
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    fused_gr_check();
    auto run = [&](int i) {
        const Set& s = sets[i % SETS];
        for (auto& x : a) { x.w_norm = s.wn; x.w_down = s.wd; x.w_up = s.wu; x.w_inject = s.wi; }
        fused_gr_read_multi(a.data(), T, xn, st);
    };
    for (int i = 0; i < 8; ++i) run(i);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, st));
    for (int i = 0; i < reps; ++i) run(i);
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    const double us = 1000.0 * ms / reps, bytes = (q8 ? 1.125 : 2.0) * 2.0 * LR * D + 2.0 * HC * D;
    // checksum on set 0
    run(0);
    CK(cudaStreamSynchronize(st));
    double cs = 0, ci = 0;
    std::vector<float> h(N), hi(HC);
    for (int t = 0; t < T; ++t) {
        CK(cudaMemcpy(h.data(), mixed[t], N * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hi.data(), inj[t], HC * 4, cudaMemcpyDeviceToHost));
        for (float v : h) cs += std::fabs(v);
        for (float v : hi) ci += v;
    }
    std::printf("T=%d  %.1f us per read  %.0f GB/s   checksum mixed %.6f inject %.6f\n", T, us, bytes / (us * 1e3), cs, ci);
    return 0;
}
