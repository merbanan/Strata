// src/prefill/moe_fp16tc.cu - see include/strata/prefill/moe_fp16tc.hpp (sm_70 FP16 tensor-core routed experts).
//
// Two kernels: the register-fragment one (expert_kernel_rf, below; the default) and the shared-memory WMMA one
// (STRATA_PREFILL_FP16TC_RF=0).  The WMMA kernel: each block computes 64 routed rows by 128 output columns. Its eight warps each own a 32 by 32 tile.
// The reduction advances by 64 values, which is one native Q2_0 weight block. WMMA m16n16k16 uses Volta's
// FP16 tensor cores with FP32 accumulation; the public WMMA interface avoids dependence on fragment lane maps.
//
// Native Q2_0 weights have a half scale and two-bit codes. Half2 arithmetic reconstructs (q - 1) * scale.
// For q8_1 activations, PRMT reconstructs the signed integer codes exactly, then half2 arithmetic multiplies
// them by the activation scale rounded to FP16. This is not the same rounding as a float multiply followed by
// one FP16 conversion, and it is not bit-identical to MMQ. The numerical regression compares both paths with
// a double-precision reference.
//
// The shared operand strides are multiples of eight halves, as WMMA requires. The same storage holds the
// accumulator tiles after the last reduction step; guarded stores exclude the next expert's rows.
#include "strata/prefill/moe_fp16tc.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::fp16tc {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill fp16tc: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int BM = 64;             // routed rows per block (2 warp row-halves of 32)
constexpr int BN = 128;            // output columns per block (4 warp column-quarters of 32)
constexpr int KS = 64;             // k per step: one Q2_0 weight block
constexpr int TS = 256;            // threads (8 warps: 2 row-halves x 4 column-quarters)
constexpr int AWMMA = 80;          // the wmma path needs a leading dimension that is a multiple of 8 halves
constexpr int WWMMA = 72;
constexpr int QB = 18;             // Q2_0 block bytes: fp16 scale + 16 code bytes
constexpr int ACTB = 144;          // block_q8_1_mmq (D4): 4 float scales + 128 int8
constexpr int Q2 = 42;             // GGML_TYPE_Q2_0

// Bit pattern -> half2 without going through the float pipe (the bits are 1024 + q in each half).
__device__ __forceinline__ __half2 h2_from_u32(uint32_t u) {
    __half2 h;
    __builtin_memcpy(&h, &u, 4);
    return h;
}

// Weight row r of the gathered matrices: GU rows are gate 0..n_ff-1 then up (at `up_off`); down rows are the down
// matrix, which the caller passes separately.
template <bool GU>
__device__ __forceinline__ const uint8_t* wbase(const Batch& b, const Geom& g, int z, int r) {
    if constexpr (GU) {
        return (r < g.n_ff) ? b.blob[z] + (size_t) r * g.gu_row
                            : b.blob[z] + g.up_off + (size_t) (r - g.n_ff) * g.gu_row;
    } else {
        return b.down[z] + (size_t) r * g.d_row;
    }
}

// NCOL weight rows (one Q2_0 block, 64 codes, each) into w_s as FP16.  256 threads cover NCOL rows x 4 quarters;
// a thread takes one row's `64/TPR` code bytes and writes the same count of half2 pairs.
template <bool GU, int WS, int NCOL>
__device__ __forceinline__ void dequant_weights(const Batch& b, const Geom& g, int z, int out_base, int k0, int tid,
                                                __half* w_s) {
    constexpr int TPR = TS / NCOL;                 // threads per row (2 at 128 cols, 4 at 64)
    constexpr int BYTES = 64 / TPR / 4;            // code bytes per thread
    constexpr int NWORD = BYTES / 4;
    constexpr int NPAIR = BYTES * 4 / 2;
    const int col = tid / TPR, sub = tid % TPR;
    const uint8_t* blk = wbase<GU>(b, g, z, out_base + col) + (size_t) (k0 / KS) * QB;
    const __half dh = *reinterpret_cast<const __half*>(blk);
    const uint8_t* qb = blk + 2 + sub * BYTES;
    const __half2 d2 = __half2half2(dh);
    const __half2 k1025 = __float2half2_rn(1025.0f);
    __half2* o = reinterpret_cast<__half2*>(w_s + (size_t) col * WS + sub * BYTES * 4);
    uint32_t words[NWORD];
#pragma unroll
    for (int w = 0; w < NWORD; ++w) {
        // Q2_0's 18-byte blocks align codes to two bytes, not necessarily four.
        const auto* q = reinterpret_cast<const uint16_t*>(qb + 4 * w);
        words[w] = (uint32_t) q[0] | ((uint32_t) q[1] << 16);
    }
#pragma unroll
    for (int k = 0; k < NPAIR; ++k) {   // pair k: codes 2k (bits 4k) and 2k+1 (bits 4k+2)
        const uint32_t bytes = words[k >> 3];
        const uint32_t t = (bytes >> (4 * (k & 7))) & 0xFu;
        const uint32_t p = (t & 3u) | ((t & 0xCu) << 14);
        o[k] = __hmul2(__hsub2(h2_from_u32(0x64006400u | p), k1025), d2);
    }
}

// BM activation rows of the layer's q8_1, 16 values per thread, as FP16.  `act_row_base` shifts the row the q8_1 is
// read from (0 for gate/up, Batch::down_act_row_base for down); the zero fill past the expert's last row keeps the
// mma from reading the next expert's activations.
template <int AS>
__device__ __forceinline__ void dequant_act(int row0, int local, int k0, int tid, const void* act, int64_t act_rows,
                                            int64_t act_row_base, __half* a_s) {
    const int m = tid >> 2, part = tid & 3;
    __half* o = a_s + (size_t) m * AS + part * 16;
    if (m >= local) {   // past the expert's rows: zero, so the mma never sees the next expert's activations
#pragma unroll
        for (int j = 0; j < 16; ++j) o[j] = __float2half(0.0f);
    } else {
        const int64_t row = (int64_t) row0 + m + act_row_base;
        const int64_t kb = k0 / 128;
        const int off = (k0 % 128) + part * 16;
        const uint8_t* base = (const uint8_t*) act + ((size_t) kb * (size_t) act_rows + (size_t) row) * ACTB;
        const float d = ((const float*) base)[off >> 5];
        const uint32_t* qs = reinterpret_cast<const uint32_t*>(base + 16 + off);
        const __half2 d2 = __float2half2_rn(d);
        const __half2 c1152 = h2_from_u32(0x64806480u);   // 1152.0h: the offset baked into the mantissa bias
        __half2* o2 = reinterpret_cast<__half2*>(o);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            // Sign-flip every code byte, then PRMT each pair into {0x64, byte} halves = 1024 + (byte ^ 128) in
            // [1024, 1279].  Subtracting 1152 leaves the signed code exactly: byte < 128 -> +byte (the XOR added
            // 128), byte >= 128 -> byte - 256.  Both operands and the difference are exact in FP16.
            const uint32_t v = qs[k] ^ 0x80808080u;
            o2[2 * k] = __hmul2(__hsub2(h2_from_u32(__byte_perm(v, 0x64646464u, 0x4140)), c1152), d2);
            o2[2 * k + 1] = __hmul2(__hsub2(h2_from_u32(__byte_perm(v, 0x64646464u, 0x4342)), c1152), d2);
        }
    }
}

// Each warp owns two row tiles and two column tiles. Stage the accumulators through shared memory because
// WMMA does not specify the fragment element map. Partial expert rows are excluded by the final store.
template <bool GU>
__global__ void __launch_bounds__(TS, 1)
expert_kernel_wmma(const Batch b, const Geom g, const int32_t* __restrict__ bounds, const void* __restrict__ act,
                   int64_t act_rows, float* __restrict__ dst, int64_t ld_dst, int64_t dst_row_base, int k_reduce) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    const int z = blockIdx.z;
    const int lo = bounds[z], hi = bounds[z + 1];
    const int row0 = lo + (int) blockIdx.y * BM;
    if (row0 >= hi) return;
    const int local = (hi - row0 < BM) ? (hi - row0) : BM;
    const int out_base = (int) blockIdx.x * BN;
    const int tid = threadIdx.x, warp = tid >> 5;
    const int rh = warp >> 2, cq = warp & 3;
    const int m_base = rh * 32, n_base = cq * 32;
    const int64_t dst_shift = dst_row_base + (GU ? b.gu_dst_row_base : 0);
    const int64_t act_shift = GU ? 0 : b.down_act_row_base;

    __shared__ __align__(32) union {
        struct { __half a[BM * AWMMA]; __half w[BN * WWMMA]; } ab;
        float c[BM * BN];
    } sm;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int c = 0; c < 2; ++c) wmma::fill_fragment(acc[r][c], 0.0f);

    for (int k0 = 0; k0 < k_reduce; k0 += KS) {
        __syncthreads();
        dequant_weights<GU, WWMMA, BN>(b, g, z, out_base, k0, tid, sm.ab.w);
        dequant_act<AWMMA>(row0, local, k0, tid, act, act_rows, act_shift, sm.ab.a);
        __syncthreads();
#pragma unroll
        for (int k16 = 0; k16 < KS / 16; ++k16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf[2];
#pragma unroll
            for (int r = 0; r < 2; ++r)
                wmma::load_matrix_sync(af[r], &sm.ab.a[(size_t) (m_base + 16 * r) * AWMMA + k16 * 16], AWMMA);
#pragma unroll
            for (int c = 0; c < 2; ++c)
                wmma::load_matrix_sync(bf[c], &sm.ab.w[(size_t) (n_base + 16 * c) * WWMMA + k16 * 16], WWMMA);
#pragma unroll
            for (int r = 0; r < 2; ++r)
#pragma unroll
                for (int c = 0; c < 2; ++c) wmma::mma_sync(acc[r][c], af[r], bf[c], acc[r][c]);
        }
    }
    __syncthreads();   // the tiles are dead: the same storage becomes the accumulator staging
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int c = 0; c < 2; ++c)
            wmma::store_matrix_sync(&sm.c[(size_t) (m_base + 16 * r) * BN + n_base + 16 * c], acc[r][c], BN,
                                    wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < BM * BN; i += TS) {
        const int grow = row0 + (i >> 7);
        if (grow < hi) dst[(dst_shift + grow) * ld_dst + out_base + (i & 127)] = sm.c[i];
    }
#endif
}

// ---- the register-fragment kernel (STRATA_PREFILL_FP16TC_RF, the default): the weights never touch shared memory.
//
// mma.m8n8k4 (Volta's native HMMA shape) splits a warp into four quad pairs (lanes 4q..4q+3 and 16+4q..16+4q+3), each
// an independent 8 x 8 x 4 product: a lane holds one row of A (4 halves), one column of B (4 halves) and 8 FP32 sums.
// A block is 64 tokens (the expert's rows) by two weight sets of 64 rows (gate/up: gate rows r and up rows r; down: two
// groups of 64 output rows), eight warps of 8 rows a set.  A k step is 128 values - one q8_1 activation block - and
// quad pair q takes its quarter [32 q, 32 q + 32): the lane decodes its row's 32 Q2_0 codes (8 bytes) straight into
// the 16 half2 of eight A fragments, scaled in FP16.  The activations are staged once a step as FP16 in shared memory
// (double-buffered, the next step's q8_1 read into registers before this step's products), and each lane reads its
// token's 32 values as eight B fragments.  The four quarters' partial sums are added by shuffles at the end.
constexpr int RT = 64;             // tokens a block
constexpr int RK = 128;            // k a step
constexpr int RAS = RK + 8;        // shared row stride in halves (8 tokens' 16-byte reads on distinct banks)

__device__ __forceinline__ void mma884(float* c, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#endif
}

// the raw bits of one lane's weight quarter: the 8 code bytes (2-byte aligned in the 18-byte block) and the scale
struct RawW { uint32_t lo, hi; __half d; };
__device__ __forceinline__ RawW load_w(const uint8_t* row, int k) {   // k: the quarter's first value (a multiple of 32)
    const uint8_t* blk = row + (size_t) (k >> 6) * QB;
    const uint16_t* q = reinterpret_cast<const uint16_t*>(blk + 2 + ((k >> 5) & 1) * 8);
    RawW r;
    r.lo = (uint32_t) __ldg(q) | ((uint32_t) __ldg(q + 1) << 16);
    r.hi = (uint32_t) __ldg(q + 2) | ((uint32_t) __ldg(q + 3) << 16);
    r.d = __ldg(reinterpret_cast<const __half*>(blk));
    return r;
}
// value 2t, 2t + 1 of the quarter as half2 (q - 1) * d: codes at bits 2i of the little-endian code stream
__device__ __forceinline__ void unpack_w(const RawW& r, uint32_t* a) {
    const __half2 d2 = __half2half2(r.d), k1025 = __float2half2_rn(1025.0f);
#pragma unroll
    for (int t = 0; t < 16; ++t) {
        const uint32_t word = t < 8 ? r.lo : r.hi;
        const uint32_t n = (word >> (4 * (t & 7))) & 0xFu;
        const uint32_t p = (n & 3u) | ((n & 0xCu) << 14);
        const __half2 h = __hmul2(__hsub2(h2_from_u32(0x64006400u | p), k1025), d2);
        a[t] = *reinterpret_cast<const uint32_t*>(&h);
    }
}

// one thread's share of a step's activations: token tid / 4, values [32 (tid % 4), +32) of the q8_1 block
struct RawA { uint4 q0, q1; float d; };
__device__ __forceinline__ RawA load_a(const void* act, int64_t act_rows, int64_t row, bool live, int kb, int part) {
    RawA r;
    if (!live) { r.q0 = make_uint4(0, 0, 0, 0); r.q1 = r.q0; r.d = 0.0f; return r; }
    const uint8_t* base = (const uint8_t*) act + ((size_t) kb * (size_t) act_rows + (size_t) row) * ACTB;
    r.d = __ldg((const float*) base + part);
    const uint4* q = reinterpret_cast<const uint4*>(base + 16 + 32 * part);
    r.q0 = __ldg(q);
    r.q1 = __ldg(q + 1);
    return r;
}
__device__ __forceinline__ void store_a(const RawA& r, __half* o) {
    const __half2 d2 = __float2half2_rn(r.d);
    const __half2 c1152 = h2_from_u32(0x64806480u);
    const uint32_t w[8] = {r.q0.x, r.q0.y, r.q0.z, r.q0.w, r.q1.x, r.q1.y, r.q1.z, r.q1.w};
    uint32_t h[16];
#pragma unroll
    for (int k = 0; k < 8; ++k) {   // (the exact signed code from the mantissa bias, as dequant_act above)
        const uint32_t v = w[k] ^ 0x80808080u;
        const __half2 x0 = __hmul2(__hsub2(h2_from_u32(__byte_perm(v, 0x64646464u, 0x4140)), c1152), d2);
        const __half2 x1 = __hmul2(__hsub2(h2_from_u32(__byte_perm(v, 0x64646464u, 0x4342)), c1152), d2);
        h[2 * k] = *reinterpret_cast<const uint32_t*>(&x0);
        h[2 * k + 1] = *reinterpret_cast<const uint32_t*>(&x1);
    }
    uint4* o4 = reinterpret_cast<uint4*>(o);
#pragma unroll
    for (int i = 0; i < 4; ++i) o4[i] = make_uint4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
}

template <bool GU>
__global__ void __launch_bounds__(TS, 1)
expert_kernel_rf(const Batch b, const Geom g, const int32_t* __restrict__ bounds, const void* __restrict__ act,
                 int64_t act_rows, float* __restrict__ dst, int64_t ld_dst, int64_t dst_row_base, int k_reduce) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    const int z = blockIdx.z;
    const int lo = bounds[z], hi = bounds[z + 1];
    const int row0 = lo + (int) blockIdx.y * RT;
    if (row0 >= hi) return;
    const int local = min(RT, hi - row0);
    const int ngrp = (local + 7) >> 3;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int qp = (lane >> 2) & 3;                       // the lane's quad pair: its k quarter
    const int j = (lane & 3) | ((lane >> 4) << 2);        // its A row / B column within the 8
    const int64_t dst_shift = dst_row_base + (GU ? b.gu_dst_row_base : 0);
    const int64_t act_shift = GU ? 0 : b.down_act_row_base;
    // the lane's weight rows: gate/up gate row r and up row n_ff + r (r in the block's 64); down rows r and r + 64 of
    // the block's 128
    const int r = (int) blockIdx.x * (GU ? 64 : 128) + 8 * warp + j;
    const uint8_t* w0 = wbase<GU>(b, g, z, r);
    const uint8_t* w1 = wbase<GU>(b, g, z, GU ? g.n_ff + r : r + 64);

    __shared__ __align__(16) __half sa[2][RT * RAS];
    float c[2][8][8];
#pragma unroll
    for (int s2 = 0; s2 < 2; ++s2)
#pragma unroll
        for (int gi = 0; gi < 8; ++gi)
#pragma unroll
            for (int i = 0; i < 8; ++i) c[s2][gi][i] = 0.0f;

    const int am = tid >> 2, apart = tid & 3;
    const bool alive = am < local;
    const int64_t arow = (int64_t) row0 + am + act_shift;
    const int nsteps = k_reduce / RK;
    RawA ra = load_a(act, act_rows, arow, alive, 0, apart);
    RawW rw0 = load_w(w0, 32 * qp), rw1 = load_w(w1, 32 * qp);
    store_a(ra, &sa[0][am * RAS + 32 * apart]);
    __syncthreads();
    for (int st = 0; st < nsteps; ++st) {
        const int buf = st & 1;
        uint32_t a0[16], a1[16];
        unpack_w(rw0, a0);
        unpack_w(rw1, a1);
        if (st + 1 < nsteps) {   // the next step's raw operands in flight under this step's products
            ra = load_a(act, act_rows, arow, alive, st + 1, apart);
            rw0 = load_w(w0, (st + 1) * RK + 32 * qp);
            rw1 = load_w(w1, (st + 1) * RK + 32 * qp);
        }
        const __half* sb = sa[buf];
#pragma unroll
        for (int gi = 0; gi < 8; ++gi) {
            if (gi < ngrp) {
                const uint4* bp = reinterpret_cast<const uint4*>(sb + (8 * gi + j) * RAS + 32 * qp);
                uint4 bv[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) bv[i] = bp[i];
                const uint32_t* bw = reinterpret_cast<const uint32_t*>(bv);
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    mma884(c[0][gi], a0[2 * kk], a0[2 * kk + 1], bw[2 * kk], bw[2 * kk + 1]);
                    mma884(c[1][gi], a1[2 * kk], a1[2 * kk + 1], bw[2 * kk], bw[2 * kk + 1]);
                }
            }
        }
        if (st + 1 < nsteps) store_a(ra, &sa[buf ^ 1][am * RAS + 32 * apart]);
        __syncthreads();
    }
    // the four quarters' sums (quad pairs differ in lane bits 2 and 3), then quad pair 0 writes its 8 x 8 tiles
#pragma unroll
    for (int s2 = 0; s2 < 2; ++s2)
#pragma unroll
        for (int gi = 0; gi < 8; ++gi)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float v = c[s2][gi][i];
                v += __shfl_xor_sync(0xffffffffu, v, 4);
                v += __shfl_xor_sync(0xffffffffu, v, 8);
                c[s2][gi][i] = v;
            }
    if (qp != 0) return;
    const int cbase0 = GU ? (int) blockIdx.x * 64 + 8 * warp : (int) blockIdx.x * 128 + 8 * warp;
    const int cbase1 = GU ? g.n_ff + cbase0 : cbase0 + 64;
#pragma unroll
    for (int gi = 0; gi < 8; ++gi) {
        if (gi >= ngrp) continue;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int rr = (lane & 1) + (i & 2) + ((lane >> 4) << 2);   // the weight row in the 8
            const int tt = 8 * gi + (i & 4) + (lane & 2) + (i & 1);      // the token in the tile
            if (tt < local) {
                float* o = dst + (dst_shift + row0 + tt) * ld_dst;
                o[cbase0 + rr] = c[0][gi][i];
                o[cbase1 + rr] = c[1][gi][i];
            }
        }
    }
#endif
}

template <bool GU>
void launch(const Batch& b, const Geom& g, const int32_t* bounds, const void* act, int64_t act_rows, float* dst,
            int64_t ld_dst, int64_t dst_row_base, void* stream) {
    if (b.n <= 0 || b.max_rows <= 0) return;
    const int k_reduce = GU ? g.n_embd : g.n_ff;
    const int out_total = GU ? 2 * g.n_ff : g.n_embd;
    const cudaStream_t s = (cudaStream_t) stream;
    // STRATA_PREFILL_FP16TC_RF=0: the shared-memory WMMA kernel instead of the register-fragment one
    static const bool rf = !(std::getenv("STRATA_PREFILL_FP16TC_RF") && std::atoi(std::getenv("STRATA_PREFILL_FP16TC_RF")) == 0);
    if (rf && k_reduce % RK == 0 && (GU ? g.n_ff % 64 == 0 : g.n_embd % 128 == 0)) {
        const dim3 grid(GU ? g.n_ff / 64 : g.n_embd / 128, (b.max_rows + RT - 1) / RT, b.n);
        expert_kernel_rf<GU><<<grid, TS, 0, s>>>(b, g, bounds, act, act_rows, dst, ld_dst, dst_row_base, k_reduce);
        ck(cudaGetLastError(), "expert_kernel_rf");
        return;
    }
    const dim3 grid(out_total / BN, (b.max_rows + BM - 1) / BM, b.n);
    expert_kernel_wmma<GU><<<grid, TS, 0, s>>>(b, g, bounds, act, act_rows, dst, ld_dst, dst_row_base, k_reduce);
    ck(cudaGetLastError(), "expert_kernel_wmma");
}

}  // namespace

bool built() { return true; }

bool available() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) return false;
    return major == 7 && minor == 0;   // retain the existing MMQ dispatch on other architectures
}

bool geom_ok(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, size_t gu_row, size_t d_row, size_t up_off,
             size_t down_off) {
    if (gu_type != Q2 || d_type != Q2) return false;
    if (n_embd <= 0 || n_ff <= 0) return false;
    // the k steps and the down reduction are whole 64-blocks; the block's BN = 128 output columns are whole too
    // (down writes n_embd columns, gate/up write 2 n_ff)
    if (n_embd % 128 != 0 || n_ff % 64 != 0) return false;
    if (gu_row != (size_t) (n_embd / 64) * QB || d_row != (size_t) (n_ff / 64) * QB) return false;
    if (up_off != gu_row * (size_t) n_ff) return false;
    if (down_off != 2 * up_off) return false;
    return true;
}

void gu(const Batch& b, const Geom& g, const int32_t* bounds, const void* xq, int64_t xq_rows, float* dst,
        int64_t ld_dst, void* stream) {
    launch<true>(b, g, bounds, xq, xq_rows, dst, ld_dst, 0, stream);
}

void down(const Batch& b, const Geom& g, const int32_t* bounds, const void* hq, int64_t hq_rows, float* dst,
          int64_t ld_dst, int64_t dst_row_base, void* stream) {
    launch<false>(b, g, bounds, hq, hq_rows, dst, ld_dst, dst_row_base, stream);
}

}  // namespace strata::prefill::fp16tc
