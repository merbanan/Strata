// src/kernels/cuda/kv_q4.cu - see include/strata/kernels/kv_q4.hpp. Q4_0 KV with Walsh-Hadamard rotation
// (from PR #21 by code-martin; KV-streaming integration and the deterministic group maximum added on merge).
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_tq.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_q4: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// Fast Walsh-Hadamard Transform for N = 256: one warp per row, 8 values per lane in registers.
// Orthonormal (scale 1/sqrt(256) = 1/16), so it is its own inverse.
__global__ void fwht256_kernel(const float* __restrict__ src, float* __restrict__ dst, int64_t n_rows, float scale) {
    constexpr int warp_size = 32;
    constexpr int N = 256;
    constexpr int el_w = N / warp_size;   // 8

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= n_rows) return;

    const float* row_src = src + r * N;
    float* row_dst = dst + r * N;

    float reg[el_w];
    const int lane = threadIdx.x;

#pragma unroll
    for (int i = 0; i < el_w; ++i) reg[i] = row_src[i * warp_size + lane] * scale;

    // the low 5 index bits live across lanes
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val = reg[j];
            const float val2 = __shfl_xor_sync(0xffffffffu, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
    // the high 3 bits across each lane's registers
#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k] = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < el_w; ++i) row_dst[i * warp_size + lane] = reg[i];
}

// One 32-value group in one warp (lane = element), ggml's q4_0: d = (the value of largest |x|) / -8,
// q = clamp(trunc(x / d + 8.5), 0, 15). Returns the scale bits; `byte` is lane t's packed byte for t < 16
// (element t in the low nibble, t + 16 in the high one). Ties in |x| resolve to the larger value in EVERY lane,
// so all lanes agree on d (a plain `a > amax` could leave lanes with opposite signs and one block two scales).
__device__ __forceinline__ uint16_t q4_group(float x, int lane, uint8_t& byte) {
    float amax = fabsf(x), mval = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float a = __shfl_xor_sync(0xffffffffu, amax, o);
        const float v = __shfl_xor_sync(0xffffffffu, mval, o);
        if (a > amax || (a == amax && v > mval)) { amax = a; mval = v; }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int q = __float2int_rz(x * id + 8.5f);
    const uint8_t qc = (uint8_t) (q < 0 ? 0 : (q > 15 ? 15 : q));
    const uint8_t qhi = __shfl_down_sync(0xffffffffu, qc, 16);
    byte = (uint8_t) (qc | (qhi << 4));
    (void) lane;
    return f16_from_f32(d);
}

__device__ __forceinline__ void q4_store(uint8_t* pool, long long row, int b, int lane, uint16_t d, uint8_t byte) {
    block_q4_0* blk = reinterpret_cast<block_q4_0*>(pool + row * (long long) sizeof(block_q4_0) * 8) + b;
    if (lane == 0) blk->d = d;
    if (lane < 16) blk->qs[lane] = byte;
}

// kv_tq.cuh: one quantized group of 32, in the format its plane takes - q4_0, TQ4, or TQ3 (V of --kv tq4v3)
struct TqGroup {
    uint16_t d;
    uint8_t byte;      // q4_0 / TQ4: lane t < 16's packed byte
    unsigned packed;   // TQ3: the group of eight's 24 bits (lanes with (t & 7) == 0); TQ6: of four ((t & 3) == 0)
    int bits;          // 4 (q4_0 / TQ4), 3, 6
};
__device__ __forceinline__ TqGroup tq_quant(float x, int lane, int tq, bool is_v) {
    TqGroup g;
    g.bits = is_v ? (tq_v3(tq) ? 3 : 4) : (tq_k6(tq) ? 6 : 4);
    g.byte = 0;
    g.packed = 0;
    if (g.bits == 3) g.d = f16_from_f32(tq3_group(x, lane, g.packed));
    else if (g.bits == 6) g.d = f16_from_f32(tq6_group(x, lane, g.packed));
    else if (tq != kTqNone) g.d = f16_from_f32(tq4_group(x, g.byte));
    else g.d = q4_group(x, lane, g.byte);
    return g;
}
__device__ __forceinline__ void tq_store(uint8_t* pool, long long row, int b, int lane, const TqGroup& g) {
    if (g.bits == 4) { q4_store(pool, row, b, lane, g.d, g.byte); return; }
    const int bb = g.bits == 3 ? kTq3BlockBytes : kTq6BlockBytes, per = g.bits == 3 ? 8 : 4;
    uint8_t* blk = pool + row * (long long) (8 * bb) + b * bb;
    if (lane == 0) { blk[0] = (uint8_t) g.d; blk[1] = (uint8_t) (g.d >> 8); }
    if ((lane & (per - 1)) == 0) {
        uint8_t* q = blk + 2 + 3 * (lane / per);
        q[0] = (uint8_t) g.packed;
        q[1] = (uint8_t) (g.packed >> 8);
        q[2] = (uint8_t) (g.packed >> 16);
    }
}

// One block = one 32-value group of one KV head of K (plane 0) or V (plane 1) for token step_idx; 32 threads.
__global__ void kv_append_q4_kernel(uint8_t* __restrict__ k_q4, uint8_t* __restrict__ v_q4,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ step,
                                    int step_stride, int planes,
                                    const float* __restrict__ kcur, const float* __restrict__ vcur,
                                    int kv_heads, int head_dim, int page_size, KvHostPools host, int tq,
                                    int v_only) {
    const int step_idx = (int) (blockIdx.z / (unsigned) planes);
    // v_only: a single-plane call on the V pool (K8V4's V half, which passes its V pool as both) - TQ3 takes V's format
    const bool is_v = (blockIdx.z % (unsigned) planes) == 1u || v_only != 0;
    const long long pos = (long long) __ldg(step + (long long) step_idx * step_stride + kStepPos);
    const int h = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
    const long long step_off = (long long) step_idx * kv_heads * head_dim;
    const float x = ((blockIdx.z % (unsigned) planes) == 1u ? vcur : kcur)[step_off + h * head_dim + b * QK4_0 + t];
    const TqGroup g = tq_quant(x, t, tq, is_v);
    const long long page = (long long) table[pos / page_size];
    uint8_t* const dst = (blockIdx.z % (unsigned) planes) == 1u ? v_q4 : k_q4;
    if (page >= 0) tq_store(dst, (page * kv_heads + h) * page_size + (pos % page_size), b, t, g);
    if (host.k_q4 != nullptr)
        tq_store((blockIdx.z % (unsigned) planes) == 1u ? host.v_q4 : host.k_q4,
                 ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size), b, t, g);
}

// The prompt path: grid (T, kv_heads, groups), K then V; also into the staging pool (identity layout) when given.
__global__ void kv_append_q4_batch_kernel(uint8_t* __restrict__ k_q4, uint8_t* __restrict__ v_q4,
                                          const int32_t* __restrict__ table, int64_t pos0,
                                          const float* __restrict__ K, const float* __restrict__ V,
                                          int kv_heads, int head_dim, int page_size, int is_v_grid, KvHostPools host,
                                          KvHostPools stage, int tq, int v_only) {
    const long long t = blockIdx.x;
    const long long pos = pos0 + t;
    const int h = blockIdx.y, b = blockIdx.z, th = threadIdx.x;
    const bool is_v = is_v_grid != 0;
    const float x = (is_v ? V : K)[t * (kv_heads * head_dim) + h * head_dim + b * QK4_0 + th];
    const TqGroup g = tq_quant(x, th, tq, is_v || v_only != 0);
    const long long page = (long long) table[pos / page_size];
    const long long row_id = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
    if (page >= 0) tq_store(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), b, th, g);
    if (host.k_q4 != nullptr) tq_store(is_v ? host.v_q4 : host.k_q4, row_id, b, th, g);
    if (stage.k_q4 != nullptr) tq_store(is_v ? stage.v_q4 : stage.k_q4, row_id, b, th, g);
}

// Gather step[kStepWidth] cells into FP16 scratch (the non-fused attention paths)
__global__ void kv_gather_q4_kernel(const uint8_t* __restrict__ k_q4, const uint8_t* __restrict__ v_q4,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                    const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                    uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch, int tq) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int blocks_per_head = head_dim / QK4_0;                        // 8
    const int bytes_per_head = blocks_per_head * sizeof(block_q4_0);    // 144
    const long long total_blocks = n_ids * kv_heads * blocks_per_head;

    const long long blk_idx = (long long) blockIdx.x * blockDim.y + threadIdx.y;
    if (blk_idx >= total_blocks) return;

    const int t = threadIdx.x;
    const long long id = blk_idx / (kv_heads * blocks_per_head);
    const int rem = (int) (blk_idx % (kv_heads * blocks_per_head));
    const int h = rem / blocks_per_head;
    const int b = rem % blocks_per_head;

    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);

    const int j = t < 16 ? t : (t - 16);
    const bool t4 = tq != kTqNone;
    float kv_;
    if (tq_k6(tq)) {   // TQ6 K (kv_tq.cuh)
        const uint8_t* kb = k_q4 + row * (long long) (blocks_per_head * kTq6BlockBytes) + b * kTq6BlockBytes;
        kv_ = c_tq6[tq6_code(kb, t)] * f32_from_f16(tq3_scale_bits(kb));
    } else {
        const block_q4_0* k_blk = reinterpret_cast<const block_q4_0*>(k_q4 + row * bytes_per_head) + b;
        const uint8_t k_byte = k_blk->qs[j];
        kv_ = tq4_val(t < 16 ? (k_byte & 0x0Fu) : (unsigned) (k_byte >> 4), t4) * f32_from_f16(k_blk->d);
    }
    float vv;
    if (tq_v3(tq)) {   // TQ3 V (kv_tq.cuh)
        const uint8_t* vb = v_q4 + row * (long long) (blocks_per_head * kTq3BlockBytes) + b * kTq3BlockBytes;
        vv = c_tq3[tq3_code(vb, t)] * f32_from_f16(tq3_scale_bits(vb));
    } else {
        const block_q4_0* v_blk = reinterpret_cast<const block_q4_0*>(v_q4 + row * bytes_per_head) + b;
        const uint8_t v_byte = v_blk->qs[j];
        vv = tq4_val(t < 16 ? (v_byte & 0x0Fu) : (unsigned) (v_byte >> 4), t4) * f32_from_f16(v_blk->d);
    }
    const long long dst_offset = ((id * kv_heads + h) * head_dim) + (b * QK4_0 + t);
    k_scratch[dst_offset] = f16_from_f32(kv_);
    v_scratch[dst_offset] = f16_from_f32(vv);
}

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard transform's size)\n", what);
        std::exit(1);
    }
}

int g_kv_tq = kTqNone;

}  // namespace

void kv_q4_set_tq(int tq) { g_kv_tq = tq; }
int kv_q4_tq() { return g_kv_tq; }

void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    if (n_rows <= 0) return;
    const int rows_per_block = 4;
    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;
    fwht256_kernel<<<dim3((unsigned) num_blocks), dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(
        src, dst, n_rows, 1.0f / 16.0f);
    check("fwht256 launch");
}

void kv_append_q4_steps(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                        int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                        void* stream, const KvHostPools* host) {
    if (n_steps <= 0) return;
    need_256(s, "kv_append_q4");
    const int planes = (v_q4 == nullptr || (v_q4 == k_q4 && vcur == kcur)) ? 1 : 2;
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) (s.head_dim / QK4_0), (unsigned) (n_steps * planes));
    kv_append_q4_kernel<<<grid, 32, 0, (cudaStream_t) stream>>>(
        k_q4, v_q4, page_table, step, step_stride, planes, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size, host ? *host : KvHostPools{}, g_kv_tq, planes == 1 && v_q4 == k_q4 ? 1 : 0);
    check("kv_append_q4 launch");
}

void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    kv_append_q4_steps(k_q4, v_q4, page_table, step, 0, 1, kcur, vcur, s, stream, host);
}

void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host, const KvHostPools* stage) {
    if (T <= 0) return;
    need_256(s, "kv_append_q4");
    const dim3 grid((unsigned) T, (unsigned) s.n_head_kv, (unsigned) (s.head_dim / QK4_0));
    cudaStream_t cs = (cudaStream_t) stream;
    const KvHostPools h = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    for (int is_v = 0; is_v < 2; ++is_v)
        kv_append_q4_batch_kernel<<<grid, 32, 0, cs>>>(k_q4, v_q4, page_table, pos0, K, V, (int) s.n_head_kv,
                                                       (int) s.head_dim, (int) s.page_size, is_v, h, st, g_kv_tq,
                                                       k_q4 == v_q4 && K == V ? 1 : 0);
    check("kv_append_q4 batch launch");
}

void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const int blocks_per_head = (int) (s.head_dim / QK4_0);
    const int64_t total_blocks = max_ids * s.n_head_kv * blocks_per_head;
    const int rows_per_block = 4;
    const unsigned num_blocks = (unsigned) ((total_blocks + rows_per_block - 1) / rows_per_block);
    kv_gather_q4_kernel<<<dim3(num_blocks), dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(
        k_q4, v_q4, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, k_scratch,
        v_scratch, g_kv_tq);
    check("kv_gather_q4 launch");
}

}  // namespace strata::kernels
