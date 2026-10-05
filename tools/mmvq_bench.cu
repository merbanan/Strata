// tools/mmvq_bench.cu - decode GEMV timing on the shapes a verify window uses.
//
// Times the engine's native_<type>_mmvq entry points (and, once added, candidate kernels) for 1/2/4 columns, with
// enough distinct weight copies cycled to defeat L2, and reports effective weight bandwidth against a plain
// streaming-read kernel on the same card.
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace strata::kernels;

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));      \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

struct Fmt {
    const char* name;
    int block_elems, block_bytes;
    std::vector<int> f16_offsets;   // fp16 scale fields set to small finite values
    void (*mmvq)(const void*, const void*, float*, int, int, int, void*);
};

static const Fmt kFmts[] = {
    {"Q3_K", 256, 110, {108}, native_q3_k_mmvq},
    {"Q4_K", 256, 144, {0, 2}, native_q4_k_mmvq},
    {"Q5_K", 256, 176, {0, 2}, native_q5_k_mmvq},
    {"Q6_K", 256, 210, {208}, native_q6_k_mmvq},
    {"IQ4_XS", 256, 136, {0}, native_iq4_xs_mmvq},
    {"Q2_0", 64, 18, {0}, native_q2_0_mmvq},
};

static const Fmt* find_fmt(const char* n) {
    for (const Fmt& f : kFmts)
        if (!std::strcmp(f.name, n)) return &f;
    return nullptr;
}

__global__ void read_kernel(const uint4* __restrict__ p, size_t n, unsigned* out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = __ldg(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) out[0] = acc;
}

// fill one weight copy: random bytes, fp16 scales in [0.001, 0.01)
static void fill_weights(std::vector<uint8_t>& h, const Fmt& f, std::mt19937& rng) {
    for (auto& b : h) b = (uint8_t) rng();
    std::uniform_real_distribution<float> u(0.001f, 0.01f);
    const size_t nb = h.size() / f.block_bytes;
    for (size_t b = 0; b < nb; ++b)
        for (int off : f.f16_offsets) {
            const __half v = __float2half(u(rng));
            std::memcpy(&h[b * f.block_bytes + off], &v, 2);
        }
}

int main(int argc, char** argv) {
    // usage: mmvq_bench [type n_out n_in]...   (default: the verify window's large shapes)
    struct Shape { std::string type; int n_out, n_in; };
    std::vector<Shape> shapes;
    for (int i = 1; i + 2 < argc; i += 3) shapes.push_back({argv[i], std::atoi(argv[i + 1]), std::atoi(argv[i + 2])});
    if (shapes.empty())
        shapes = {{"Q3_K", 10240, 2560}, {"Q3_K", 6144, 2560}, {"IQ4_XS", 10240, 2560}, {"Q4_K", 10240, 2560},
                  {"Q3_K", 12288, 2560}, {"IQ4_XS", 2560, 4096}, {"Q4_K", 2560, 4096}, {"Q5_K", 2560, 4096},
                  {"Q6_K", 2560, 4096}, {"Q5_K", 248320, 2560}};

    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    {   // streaming-read reference over 512 MiB
        const size_t bytes = 512ull << 20;
        uint4* p;
        unsigned* o;
        CK(cudaMalloc(&p, bytes));
        CK(cudaMalloc(&o, 4));
        CK(cudaMemset(p, 1, bytes));
        int sms = 0;
        CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
        for (int warm = 0; warm < 3; ++warm) read_kernel<<<sms * 8, 256, 0, s>>>(p, bytes / 16, o);
        CK(cudaEventRecord(e0, s));
        for (int r = 0; r < 10; ++r) read_kernel<<<sms * 8, 256, 0, s>>>(p, bytes / 16, o);
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        std::printf("streaming read: %.1f GB/s\n\n", 10.0 * bytes / (ms * 1e6));
        CK(cudaFree(p));
        CK(cudaFree(o));
    }

    std::mt19937 rng(1234);
    std::printf("%-7s %7s %6s  %10s %10s %10s   (us per call / GB/s of weights)\n", "type", "n_out", "n_in", "ncols=1",
                "ncols=2", "ncols=4");
    for (const Shape& sh : shapes) {
        const Fmt* f = find_fmt(sh.type.c_str());
        if (!f) {
            std::fprintf(stderr, "unknown type %s\n", sh.type.c_str());
            return 2;
        }
        const size_t wbytes = (size_t) sh.n_out * (sh.n_in / f->block_elems) * f->block_bytes;
        const int copies = (int) std::max<size_t>(1, std::min<size_t>(16, (64ull << 20) / wbytes + 1));
        std::vector<void*> w(copies);
        std::vector<uint8_t> h(wbytes);
        for (int c = 0; c < copies; ++c) {
            fill_weights(h, *f, rng);
            CK(cudaMalloc(&w[c], wbytes));
            CK(cudaMemcpy(w[c], h.data(), wbytes, cudaMemcpyHostToDevice));
        }
        const int maxc = 8;
        float *x, *y;
        void* xq;
        CK(cudaMalloc(&x, sizeof(float) * sh.n_in * maxc));
        CK(cudaMalloc(&y, sizeof(float) * sh.n_out * maxc));
        CK(cudaMalloc(&xq, native_q8_1_bytes(sh.n_in, maxc)));
        std::vector<float> hx((size_t) sh.n_in * maxc);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : hx) v = nd(rng);
        CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        native_quantize_q8_1(x, xq, sh.n_in, maxc, s);
        std::vector<float> ref((size_t) sh.n_out * 4), got((size_t) sh.n_out * 4);
        for (int mode : {0, 16, 32}) {
            native_mmvq_set_group(mode);
            std::printf("%-7s %7d %6d g%-2d", f->name, sh.n_out, sh.n_in, mode);
            for (int nc : {1, 2, 4}) {
                const int reps = std::min(400, std::max(20, (int) (2e9 / wbytes)));
                for (int r = 0; r < 5; ++r) f->mmvq(w[r % copies], xq, y, sh.n_in, sh.n_out, nc, s);
                CK(cudaEventRecord(e0, s));
                for (int r = 0; r < reps; ++r) f->mmvq(w[r % copies], xq, y, sh.n_in, sh.n_out, nc, s);
                CK(cudaEventRecord(e1, s));
                CK(cudaEventSynchronize(e1));
                float ms;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                const double us = 1000.0 * ms / reps;
                std::printf(" %7.1f/%4.0f", us, wbytes / (us * 1e3));
                if (nc == 4) {   // accuracy of the 4-column result on copy 0 against the exact layout
                    f->mmvq(w[0], xq, y, sh.n_in, sh.n_out, nc, s);
                    CK(cudaMemcpyAsync(mode ? got.data() : ref.data(), y, got.size() * 4, cudaMemcpyDeviceToHost, s));
                    CK(cudaStreamSynchronize(s));
                    if (mode) {
                        double md = 0, mr = 0;
                        for (size_t i = 0; i < got.size(); ++i) {
                            md = std::max(md, (double) std::fabs(got[i] - ref[i]));
                            mr = std::max(mr, (double) std::fabs(ref[i]));
                        }
                        std::printf("   maxdiff %.2e (max|y| %.2e)", md, mr);
                    }
                }
            }
            std::printf("\n");
        }
        native_mmvq_set_group(0);
        for (void* p : w) CK(cudaFree(p));
        CK(cudaFree(x));
        CK(cudaFree(y));
        CK(cudaFree(xq));
    }
    return 0;
}
