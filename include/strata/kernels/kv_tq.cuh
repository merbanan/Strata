// include/strata/kernels/kv_tq.cuh - TurboQuant-style K/V codes for the QSA layers (`--kv tq4`, `--kv tq4v3`).
//
// Ported from project-maya's hy3 branch (2a1fde5, glm_gqa.cu): a rotated row's coordinates are close to a Gaussian,
// so each takes the nearest level of the Lloyd-Max codebook of a unit Gaussian (MSE 0.0095 at 4 bits, 0.0346 at 3)
// instead of a uniform grid, and the scale is the least-squares fit of the values to their levels.  No QJL residual.
//
// Strata's version keeps kv_q4.hpp's rotation (the 256-point Walsh-Hadamard transform of K, V and q; the output
// rotated back) and its 32-value groups with one fp16 scale each, where maya keeps one f32 scale per 128-value row:
//   - TQ4 block = block_q4_0's bytes (fp16 scale + 16 code bytes, element j in the low nibble of byte j, j + 16 in the
//     high one), the codes read through c_tq4 instead of (q - 8): every q4_0 pool, copy and stream path is unchanged;
//   - TQ3 block = fp16 scale + 12 code bytes (eight 3-bit codes in each 3 bytes, elements 8g..8g+7 in bytes 3g..3g+2,
//     element 8g+i at bit 3i): 14 B per 32 values, 112 B per head.
// --kv tq4: K and V TQ4, 576 B per cell (q4_0's size).  --kv tq4v3: K TQ4, V TQ3, 512 B per cell (k8v4: 816).
#pragma once

#include <cstdint>

namespace strata::kernels {

inline constexpr int kTqBlock = 32;
inline constexpr int kTq3BlockBytes = 2 + 12;   // fp16 scale + 32 x 3 bits

/// The K/V code formats of a q4-family pool (QsaAttnPools::tq, QsaState::kv_tq): 0 q4_0, 1 TQ4 K and V, 2 TQ4 K + TQ3 V
enum KvTq : int { kTqNone = 0, kTqK4V4 = 1, kTqK4V3 = 2 };

/// Bytes of one head's V row (head_dim values) in the pool: q4_0 / TQ4 144 (head_dim 256), TQ3 112.
inline uint64_t kv_tq_v_bytes_per_head(int head_dim, int tq) {
    return tq == kTqK4V3 ? (uint64_t) (head_dim / kTqBlock) * kTq3BlockBytes : (uint64_t) (head_dim / kTqBlock) * 18;
}

#if defined(__CUDACC__)
// the Lloyd-Max levels of a unit Gaussian (maya's c_cb4 / c_cb3)
static __constant__ float c_tq4[16] = {-2.732590f, -2.069017f, -1.618046f, -1.256231f, -0.942340f, -0.656759f,
                                       -0.388048f, -0.128395f, 0.128395f,  0.388048f,  0.656759f,  0.942340f,
                                       1.256231f,  1.618046f,  2.069017f,  2.732590f};
static __constant__ float c_tq3[8] = {-2.151946f, -1.343909f, -0.756005f, -0.245094f,
                                      0.245094f,  0.756005f,  1.343909f,  2.151946f};
// the levels as int8 for the prompt path's integer tiles: round(level * kTqI8), scale / kTqI8 (an error of at most
// 1/93 of a level step - below 1% of the 4-bit rounding itself)
inline constexpr float kTqI8 = 46.0f;   // 2.73259 * 46 = 125.7
static __constant__ int8_t c_tq4_i8[16] = {-126, -95, -74, -58, -43, -30, -18, -6, 6, 18, 30, 43, 58, 74, 95, 126};
static __constant__ int8_t c_tq3_i8[8] = {-99, -62, -35, -11, 11, 35, 62, 99};

/// A q4-family nibble's value in units of the block scale: q4_0 (q - 8) or the TQ4 level.
__device__ __forceinline__ float tq4_val(unsigned q, bool tq) { return tq ? c_tq4[q] : (float) ((int) q - 8); }
__device__ __forceinline__ int tq4_i8(unsigned q, bool tq) { return tq ? (int) c_tq4_i8[q] : (int) q - 8; }

/// The 3-bit code of element e (0..31) of a TQ3 block.
__device__ __forceinline__ unsigned tq3_code(const uint8_t* blk, int e) {
    const uint8_t* g = blk + 2 + 3 * (e >> 3);
    const unsigned u = (unsigned) g[0] | ((unsigned) g[1] << 8) | ((unsigned) g[2] << 16);
    return (u >> (3 * (e & 7))) & 7u;
}
/// The fp16 scale bits of a TQ3 block.
__device__ __forceinline__ uint16_t tq3_scale_bits(const uint8_t* blk) {
    return (uint16_t) ((unsigned) blk[0] | ((unsigned) blk[1] << 8));
}
/// Eight consecutive values (elements e0..e0+7, e0 a multiple of 8) of a TQ3 block, scaled.
__device__ __forceinline__ void tq3_load8(const uint8_t* blk, int e0, float d, float* out) {
    const uint8_t* g = blk + 2 + 3 * (e0 >> 3);
    const unsigned u = (unsigned) g[0] | ((unsigned) g[1] << 8) | ((unsigned) g[2] << 16);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = c_tq3[(u >> (3 * i)) & 7u] * d;
}

__device__ __forceinline__ float tq_wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

/// One 32-value group in one warp (lane = element), TQ4: the nearest level of x * sqrt(32 / sum x^2), the scale the
/// least-squares fit sum(x c) / sum(c c).  Returns the fp16 scale; `byte` as q4_group's (lane t < 16: element t in the
/// low nibble, t + 16 in the high one).  Every lane computes the same sums (xor reductions), so all agree on the scale.
__device__ __forceinline__ float tq4_group(float x, uint8_t& byte) {
    const float ss = tq_wsum(x * x);
    const float y = ss > 0.0f ? x * sqrtf((float) kTqBlock / ss) : 0.0f;
    unsigned code = 0;
#pragma unroll
    for (int i = 0; i < 15; ++i) code += y > 0.5f * (c_tq4[i] + c_tq4[i + 1]);
    const float c = c_tq4[code];
    const float num = tq_wsum(x * c), den = tq_wsum(c * c);
    const unsigned hi = __shfl_down_sync(0xffffffffu, code, 16);
    byte = (uint8_t) (code | (hi << 4));
    return den > 0.0f ? num / den : 0.0f;
}

/// TQ3: the same with the 3-bit levels.  Returns the scale; `packed` is the group of eight's 24 bits, valid in lanes
/// with (lane & 7) == 0 (elements lane..lane+7, element lane+i at bit 3i).
__device__ __forceinline__ float tq3_group(float x, int lane, unsigned& packed) {
    const float ss = tq_wsum(x * x);
    const float y = ss > 0.0f ? x * sqrtf((float) kTqBlock / ss) : 0.0f;
    unsigned code = 0;
#pragma unroll
    for (int i = 0; i < 7; ++i) code += y > 0.5f * (c_tq3[i] + c_tq3[i + 1]);
    const float c = c_tq3[code];
    const float num = tq_wsum(x * c), den = tq_wsum(c * c);
    unsigned u = code << (3 * (lane & 7));
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) u |= __shfl_xor_sync(0xffffffffu, u, o);
    packed = u;
    return den > 0.0f ? num / den : 0.0f;
}
#endif

}  // namespace strata::kernels
