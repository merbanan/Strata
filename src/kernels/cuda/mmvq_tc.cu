// mmvq_tc.cu - the dense GGUF K-quant matvec of a verify window on int8 tensor cores (sm_75+), STRATA_MMVQ_TC=1.
//
// native_mmvq's dp4a kernels are bandwidth-bound at one column but compute-bound from two columns on: on an RTX 2060
// SUPER a Q5_K 2560x12288 call takes 55.6 us at one column (87% of the read roofline) and 97.5 us at four (50%).  Here
// a warp computes 8 weight rows x up to 8 columns with `mma.sync.m8n8k16.s8`: the weight rows are the A operand, the
// columns' Q8_1 codes the B operand, so a window of 2-8 tokens reads the weights once at (close to) the one-column cost.
//
// No repack: the GGUF blocks are read in place.  A lane group of 4 (one row) reads the 32 contiguous qs bytes of a
// 64-value chunk (8 bytes a lane, one 32-byte sector a row); the low nibbles are sub-block 2c, the high ones 2c+1.
// The MMA's k slots are permuted to match those loads (slot 4*(lane%4)+j of step t holds value 8*(lane%4)+4t+j of the
// sub-block, for the weights and the activations alike - a dot product does not care about the order), so every
// 32-value sub-block is two MMAs into its own int32 accumulator, and its K-quant scale and min are applied in fp32:
//     y[row][col] += dx*(d*sc*sum(q*xq) - dmin*m*sum(xq))   (dx: the column's Q8_1 scale for that sub-block)
// The float summation order differs from native_mmvq's, so outputs are not bitwise equal (relative error ~1e-6).
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace strata::kernels {
namespace {

struct TcQ81 { half2 ds; int8_t qs[32]; };
struct TcQ4K { half2 dm; uint8_t scales[12]; uint8_t qs[128]; };
struct TcQ5K { half2 dm; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
static_assert(sizeof(TcQ81) == 36 && sizeof(TcQ4K) == 144 && sizeof(TcQ5K) == 176, "GGUF block sizes");

constexpr int TC_WARPS = 4;   // warps a block, split along K; each block owns 8 rows

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int a, int b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
#else
    (void) c0; (void) c1; (void) a; (void) b;
#endif
}

// one sub-block's scale and min from the 12 packed bytes (get_scale_min_k4), j = 0..7
__device__ __forceinline__ void k4_scale(const uint32_t w0, const uint32_t w1, const uint32_t w2, int j, float& sc,
                                         float& mn) {
    const uint32_t ws[3] = {w0, w1, w2};
    const uint8_t* q = reinterpret_cast<const uint8_t*>(ws);
    int a, b;
    if (j < 4) {
        a = q[j] & 63;
        b = q[j + 4] & 63;
    } else {
        a = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        b = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
    sc = float(a);
    mn = float(b);
}

// RT row tiles of 8 a warp (8*RT rows share every activation fragment and scale), NW warps a block split along K
template <bool Q5, int RT, int NW>
__launch_bounds__(NW * 32)
__global__ void kq_tc_kernel(const uint8_t* __restrict__ w, const TcQ81* __restrict__ x, float* __restrict__ y,
                             int n_in, int n_out, int ncols) {
    constexpr int BB = Q5 ? 176 : 144;   // block bytes
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r = lane >> 2, q = lane & 3;
    const int bpr = n_in / 256, nb = n_in / 32;          // superblocks a row, Q8_1 blocks a column
    const int row0 = blockIdx.x * 8 * RT + r;            // this lane's row in tile 0 (tile i: + 8 i)
    const bool bcol_ok = r < ncols;                      // the column this lane feeds into B is r
    const int c0 = 2 * q, c1 = 2 * q + 1;                // the columns of this lane's two outputs
    const TcQ81* xb = x + (size_t) (bcol_ok ? r : 0) * nb;
    const TcQ81* x0 = x + (size_t) (c0 < ncols ? c0 : 0) * nb;
    const TcQ81* x1 = x + (size_t) (c1 < ncols ? c1 : 0) * nb;
    float acc[RT][2] = {};
    const uint8_t* wr[RT];
    bool rok[RT];
#pragma unroll
    for (int i = 0; i < RT; ++i) {
        rok[i] = row0 + 8 * i < n_out;
        wr[i] = w + (size_t) (rok[i] ? row0 + 8 * i : 0) * bpr * BB;
    }
    for (int sb = warp; sb < bpr; sb += NW) {
        float2 dm[RT];
        uint32_t s0[RT], s1[RT], s2[RT];
        uint2 qh2[RT];
#pragma unroll
        for (int i = 0; i < RT; ++i) {
            const uint8_t* b = wr[i] + (size_t) sb * BB;
            dm[i] = __half22float2(*reinterpret_cast<const half2*>(b));
            s0[i] = *reinterpret_cast<const uint32_t*>(b + 4);
            s1[i] = *reinterpret_cast<const uint32_t*>(b + 8);
            s2[i] = *reinterpret_cast<const uint32_t*>(b + 12);
            qh2[i] = Q5 ? *reinterpret_cast<const uint2*>(b + 16 + 8 * q) : make_uint2(0, 0);
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            uint2 v[RT];
#pragma unroll
            for (int i = 0; i < RT; ++i)
                v[i] = rok[i] ? *reinterpret_cast<const uint2*>(wr[i] + (size_t) sb * BB + (Q5 ? 48 : 16) + 32 * c + 8 * q)
                              : make_uint2(0, 0);
#pragma unroll
            for (int h = 0; h < 2; ++h) {   // h 0: sub-block 2c (low nibbles), 1: 2c+1 (high)
                const int S = 2 * c + h;
                const int kb = sb * 8 + S;   // the Q8_1 block of this sub-block
                int bx0 = 0, bx1 = 0;
                if (bcol_ok) {
                    const int* xq = reinterpret_cast<const int*>(xb[kb].qs + 8 * q);   // 4-byte aligned (36-byte blocks)
                    bx0 = __ldg(xq);
                    bx1 = __ldg(xq + 1);
                }
                const float dx0 = c0 < ncols ? __low2float(x0[kb].ds) : 0.f;
                const float dx1 = c1 < ncols ? __low2float(x1[kb].ds) : 0.f;
                // the min term takes the column's int8 codes' sum (as llama.cpp's vec dot does), not the stored sum of
                // the original floats: with the codes in both terms their quantization errors cancel as the weights'
                // d*sc*q and dmin*m do.  One all-ones A operand gives every lane the sums of exactly its two columns.
                int u0 = 0, u1 = 0;
                mma_s8(u0, u1, 0x01010101, bx0);
                mma_s8(u0, u1, 0x01010101, bx1);
#pragma unroll
                for (int i = 0; i < RT; ++i) {
                    const int sh = 4 * h;
                    int a0 = (int) ((v[i].x >> sh) & 0x0F0F0F0Fu), a1 = (int) ((v[i].y >> sh) & 0x0F0F0F0Fu);
                    if (Q5) {
                        a0 |= (int) (((qh2[i].x >> S) & 0x01010101u) << 4);
                        a1 |= (int) (((qh2[i].y >> S) & 0x01010101u) << 4);
                    }
                    int t0 = 0, t1 = 0;
                    mma_s8(t0, t1, a0, bx0);
                    mma_s8(t0, t1, a1, bx1);
                    float sc, mn;
                    k4_scale(s0[i], s1[i], s2[i], S, sc, mn);
                    const float dsc = dm[i].x * sc, dmn = dm[i].y * mn;
                    acc[i][0] += dx0 * (dsc * (float) t0 - dmn * (float) u0);
                    acc[i][1] += dx1 * (dsc * (float) t1 - dmn * (float) u1);
                }
            }
        }
    }
    __shared__ float part[NW][RT][32][2];
#pragma unroll
    for (int i = 0; i < RT; ++i) {
        part[warp][i][lane][0] = acc[i][0];
        part[warp][i][lane][1] = acc[i][1];
    }
    __syncthreads();
    // warp k reduces tile k (and k + NW, ...)
    for (int i = warp; i < RT; i += NW) {
        float a0 = 0.f, a1 = 0.f;
#pragma unroll
        for (int k = 0; k < NW; ++k) {
            a0 += part[k][i][lane][0];
            a1 += part[k][i][lane][1];
        }
        const int row = row0 + 8 * i;
        if (row >= n_out) continue;
        if (c0 < ncols) y[(size_t) c0 * n_out + row] = a0;
        if (c1 < ncols) y[(size_t) c1 * n_out + row] = a1;
    }
}

// Q6_K (210-byte blocks: only 2-byte aligned, so 32-bit words are read as two halves).  Value l + 32 g of half n is
// ((ql nibble) | (qh 2 bits) << 4) - 32 with one int8 scale per 16 values; every 16-value scale group is one MMA:
// slot 4*(lane%4)+j holds l = 16 u + 4 (lane%4) + j, for the weights and for the column's Q8_1 codes alike.
__device__ __forceinline__ uint32_t ld32_a2(const uint8_t* p) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
    return (uint32_t) h[0] | ((uint32_t) h[1] << 16);
}

template <int NW>
__launch_bounds__(NW * 32)
__global__ void q6_tc_kernel(const uint8_t* __restrict__ w, const TcQ81* __restrict__ x, float* __restrict__ y,
                             int n_in, int n_out, int ncols) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r = lane >> 2, q = lane & 3;
    const int bpr = n_in / 256, nb = n_in / 32;
    const int row = blockIdx.x * 8 + r;
    const bool row_ok = row < n_out;
    const bool bcol_ok = r < ncols;
    const int c0 = 2 * q, c1 = 2 * q + 1;
    const TcQ81* xb = x + (size_t) (bcol_ok ? r : 0) * nb;
    const TcQ81* x0 = x + (size_t) (c0 < ncols ? c0 : 0) * nb;
    const TcQ81* x1 = x + (size_t) (c1 < ncols ? c1 : 0) * nb;
    const uint8_t* wr = w + (size_t) (row_ok ? row : 0) * bpr * 210;
    float acc0 = 0.f, acc1 = 0.f;
    for (int sb = warp; sb < bpr; sb += NW) {
        const uint8_t* b = wr + (size_t) sb * 210;
        const float d = row_ok ? __half2float(*reinterpret_cast<const half*>(b + 208)) : 0.f;
        int8_t sc[16];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint16_t v = reinterpret_cast<const uint16_t*>(b + 192)[k];
            sc[2 * k] = (int8_t) (v & 0xFF);
            sc[2 * k + 1] = (int8_t) (v >> 8);
        }
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int o = 16 * u + 4 * q;
                const uint32_t la = row_ok ? ld32_a2(b + 64 * n + o) : 0u;        // ql[l]      (g 0 low, g 2 high)
                const uint32_t lb = row_ok ? ld32_a2(b + 64 * n + 32 + o) : 0u;   // ql[l + 32] (g 1 low, g 3 high)
                const uint32_t hq = row_ok ? ld32_a2(b + 128 + 32 * n + o) : 0u;  // qh[l]
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const uint32_t lo = (g & 1) ? lb : la;
                    const uint32_t nib = (g < 2 ? lo : lo >> 4) & 0x0F0F0F0Fu;
                    const uint32_t hb = ((hq >> (2 * g)) & 0x03030303u) << 4;
                    const int a = (int) __vsub4(nib | hb, 0x20202020u);   // -32..31 a byte
                    const int kb = sb * 8 + 4 * n + g;                     // the Q8_1 block: values 32 g .. of half n
                    const int bx = bcol_ok ? __ldg(reinterpret_cast<const int*>(xb[kb].qs + o)) : 0;
                    int t0 = 0, t1 = 0;
                    mma_s8(t0, t1, a, bx);
                    const float ds = d * (float) sc[8 * n + 2 * g + u];
                    if (c0 < ncols) acc0 += ds * __low2float(x0[kb].ds) * (float) t0;
                    if (c1 < ncols) acc1 += ds * __low2float(x1[kb].ds) * (float) t1;
                }
            }
        }
    }
    __shared__ float part[NW][32][2];
    part[warp][lane][0] = acc0;
    part[warp][lane][1] = acc1;
    __syncthreads();
    if (warp != 0) return;
#pragma unroll
    for (int k = 1; k < NW; ++k) {
        acc0 += part[k][lane][0];
        acc1 += part[k][lane][1];
    }
    if (!row_ok) return;
    if (c0 < ncols) y[(size_t) c0 * n_out + row] = acc0;
    if (c1 < ncols) y[(size_t) c1 * n_out + row] = acc1;
}

template <bool Q5, int RT, int NW>
void tc_launch(const uint8_t* w, const TcQ81* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const dim3 grid((unsigned) ((n_out + 8 * RT - 1) / (8 * RT)));
    kq_tc_kernel<Q5, RT, NW><<<grid, NW * 32, 0, s>>>(w, x, y, n_in, n_out, ncols);
}

int tc_mode() {
    static const int m = [] { const char* v = std::getenv("STRATA_MMVQ_TC"); return v ? std::atoi(v) : 0; }();
    return m;
}
int tc_min_cols() {
    static const int m = [] { const char* v = std::getenv("STRATA_MMVQ_TC_MIN"); return v ? std::atoi(v) : 2; }();
    return m;
}
bool tc_arch_ok() {
    static int ok[16] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return false;
    if (ok[dev] == 0) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        ok[dev] = (major * 10 + minor >= 75) ? 1 : -1;
    }
    return ok[dev] > 0;
}

}  // namespace

bool native_mmvq_tc_supported(int ggml_type, int ncols, int n_in) {
#if defined(__HIPCC__)
    (void) ggml_type; (void) ncols; (void) n_in;
    return false;
#endif
    return (ggml_type == 12 || ggml_type == 13 || ggml_type == 14) && ncols >= 1 && ncols <= 8 && n_in % 256 == 0 && tc_arch_ok();
}

bool native_mmvq_tc(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                    void* stream, bool force) {
    if (!force && (tc_mode() == 0 || ncols < tc_min_cols())) return false;
    if (!native_mmvq_tc_supported(ggml_type, ncols, n_in)) return false;
    if (!force) {
        // diagnostics: STRATA_MMVQ_TC_ONLY=head (the vocabulary projection only) / body (everything else);
        // Q6_K only with STRATA_MMVQ_TC_Q6=1 (newer kernel)
        static const int only = [] {
            const char* v = std::getenv("STRATA_MMVQ_TC_ONLY");
            return v == nullptr ? 0 : v[0] == 'h' ? 1 : v[0] == 'b' ? 2 : 0;
        }();
        static const bool q6 = [] { const char* v = std::getenv("STRATA_MMVQ_TC_Q6"); return v && v[0] == '1'; }();
        const bool head = n_out >= 100000;
        if ((only == 1 && !head) || (only == 2 && head)) return false;
        if (ggml_type == 14 && !q6) return false;
    }
    const auto s = static_cast<cudaStream_t>(stream);
    const auto* w = static_cast<const uint8_t*>(weights);
    const auto* x = static_cast<const TcQ81*>(x_q8_1);
    // STRATA_MMVQ_TC_CFG=RT*10+NW (tuning): rows tiles a warp x warps a block; default by shape
    static const int cfg_env = [] { const char* v = std::getenv("STRATA_MMVQ_TC_CFG"); return v ? std::atoi(v) : 0; }();
    // measured on an RTX 2060 SUPER (mmvq_tc_bench, bench/2026-10-13-densetc): one 8-row tile x 8 warps along K up to
    // 4 columns, two tiles x 4 warps beyond; a long reduction (n_in >= 6144) at 2-3 columns keeps native_mmvq's IL
    if (!cfg_env && !force && n_in >= 6144 && ncols <= 3) return false;
    int cfg = cfg_env ? cfg_env : (ncols <= 4 ? 18 : 24);
    if (ggml_type == 14) {   // Q6_K: one 8-row tile a warp; STRATA_MMVQ_TC_Q6NW: warps along K (default 8)
        static const int nw = [] { const char* v = std::getenv("STRATA_MMVQ_TC_Q6NW"); return v ? std::atoi(v) : 8; }();
        const dim3 grid((unsigned) ((n_out + 7) / 8));
        if (nw == 4) q6_tc_kernel<4><<<grid, 128, 0, s>>>(w, x, y, n_in, n_out, ncols);
        else if (nw == 2) q6_tc_kernel<2><<<grid, 64, 0, s>>>(w, x, y, n_in, n_out, ncols);
        else q6_tc_kernel<8><<<grid, 256, 0, s>>>(w, x, y, n_in, n_out, ncols);
        return true;
    }
    const bool q5 = ggml_type == 13;
#define TC_CASE(RT, NW) case RT * 10 + NW: \
        if (q5) tc_launch<true, RT, NW>(w, x, y, n_in, n_out, ncols, s); else tc_launch<false, RT, NW>(w, x, y, n_in, n_out, ncols, s); break;
    switch (cfg) {
        TC_CASE(1, 4) TC_CASE(2, 4) TC_CASE(4, 4) TC_CASE(1, 8) TC_CASE(2, 8) TC_CASE(4, 2) TC_CASE(2, 2) TC_CASE(4, 8)
        default: if (q5) tc_launch<true, 2, 4>(w, x, y, n_in, n_out, ncols, s); else tc_launch<false, 2, 4>(w, x, y, n_in, n_out, ncols, s);
    }
#undef TC_CASE
    return true;
}

}  // namespace strata::kernels
