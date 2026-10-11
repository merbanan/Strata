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
//     element 8g+i at bit 3i): 14 B per 32 values, 112 B per head;
//   - TQ6 block (K only) = fp16 scale + 24 code bytes (four 6-bit codes in each 3 bytes, elements 4g..4g+3 in bytes
//     3g..3g+2, element 4g+i at bit 6i): 26 B per 32 values, 208 B per head.
// Bytes per cell (2 KV heads of 256): tq4 576 (q4_0's), tq4v3 512, tq6v4 704, tq6v3 640; k8v4 816.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define STRATA_TQ_HD __host__ __device__
#else
#define STRATA_TQ_HD
#endif

namespace strata::kernels {

inline constexpr int kTqBlock = 32;
inline constexpr int kTq3BlockBytes = 2 + 12;   // fp16 scale + 32 x 3 bits

inline constexpr int kTq6BlockBytes = 2 + 24;   // fp16 scale + 32 x 6 bits

/// The K/V code formats of a q4-family pool (kv_q4_set_tq, QsaAttnPools::tq): 0 q4_0, then K / V bits
enum KvTq : int { kTqNone = 0, kTqK4V4 = 1, kTqK4V3 = 2, kTqK6V4 = 3, kTqK6V3 = 4 };
STRATA_TQ_HD inline bool tq_k6(int tq) { return tq == kTqK6V4 || tq == kTqK6V3; }
STRATA_TQ_HD inline bool tq_v3(int tq) { return tq == kTqK4V3 || tq == kTqK6V3; }

/// Bytes of one head's K row (head_dim values) in the pool: q4_0 / TQ4 144 (head_dim 256), TQ6 208.
inline uint64_t kv_tq_k_bytes_per_head(int head_dim, int tq) {
    return (uint64_t) (head_dim / kTqBlock) * (tq_k6(tq) ? kTq6BlockBytes : 18);
}
/// Bytes of one head's V row in the pool: q4_0 / TQ4 144, TQ3 112.
inline uint64_t kv_tq_v_bytes_per_head(int head_dim, int tq) {
    return (uint64_t) (head_dim / kTqBlock) * (tq_v3(tq) ? kTq3BlockBytes : 18);
}

#if defined(__CUDACC__)
// the Lloyd-Max levels of a unit Gaussian (maya's c_cb4 / c_cb3)
static __constant__ float c_tq4[16] = {-2.732590f, -2.069017f, -1.618046f, -1.256231f, -0.942340f, -0.656759f,
                                       -0.388048f, -0.128395f, 0.128395f,  0.388048f,  0.656759f,  0.942340f,
                                       1.256231f,  1.618046f,  2.069017f,  2.732590f};
static __constant__ float c_tq3[8] = {-2.151946f, -1.343909f, -0.756005f, -0.245094f,
                                      0.245094f,  0.756005f,  1.343909f,  2.151946f};
static __constant__ float c_tq6[64] = {
    -3.744101f, -3.240437f, -2.917407f, -2.672274f, -2.471305f, -2.298981f, -2.146810f, -2.009611f, -1.883977f,
    -1.767542f, -1.658589f, -1.555831f, -1.458276f, -1.365141f, -1.275794f, -1.189720f, -1.106488f, -1.025736f,
    -0.947155f, -0.870477f, -0.795468f, -0.721922f, -0.649656f, -0.578503f, -0.508314f, -0.438950f, -0.370283f,
    -0.302193f, -0.234567f, -0.167297f, -0.100278f, -0.033410f, 0.033410f,  0.100278f,  0.167297f,  0.234567f,
    0.302193f,  0.370283f,  0.438950f,  0.508314f,  0.578503f,  0.649656f,  0.721922f,  0.795468f,  0.870477f,
    0.947155f,  1.025736f,  1.106488f,  1.189720f,  1.275794f,  1.365141f,  1.458276f,  1.555831f,  1.658589f,
    1.767542f,  1.883977f,  2.009611f,  2.146810f,  2.298981f,  2.471305f,  2.672274f,  2.917407f,  3.240437f,
    3.744101f};
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
/// Eight consecutive values (e0 a multiple of 8) of a TQ6 block, scaled.
__device__ __forceinline__ void tq6_load8(const uint8_t* blk, int e0, float d, float* out) {
    const uint8_t* g = blk + 2 + 3 * (e0 >> 2);   // two groups of four: bytes 6 * (e0 / 8) ..
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const unsigned u = (unsigned) g[3 * h] | ((unsigned) g[3 * h + 1] << 8) | ((unsigned) g[3 * h + 2] << 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) out[4 * h + i] = c_tq6[(u >> (6 * i)) & 63u] * d;
    }
}
/// The 6-bit code of element e of a TQ6 block.
__device__ __forceinline__ unsigned tq6_code(const uint8_t* blk, int e) {
    const uint8_t* g = blk + 2 + 3 * (e >> 2);
    const unsigned u = (unsigned) g[0] | ((unsigned) g[1] << 8) | ((unsigned) g[2] << 16);
    return (u >> (6 * (e & 3))) & 63u;
}
/// The fp16 scale bits of a TQ3 or TQ6 block.
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

/// TQ6: the same with the 64 levels.  `packed`: the group of four's 24 bits, valid in lanes with (lane & 3) == 0.
__device__ __forceinline__ float tq6_group(float x, int lane, unsigned& packed) {
    const float ss = tq_wsum(x * x);
    const float y = ss > 0.0f ? x * sqrtf((float) kTqBlock / ss) : 0.0f;
    // binary search over the 63 midpoints (sorted levels)
    unsigned code = 0;
#pragma unroll
    for (int step = 32; step > 0; step >>= 1)
        if (code + step <= 63u && y > 0.5f * (c_tq6[code + step - 1] + c_tq6[code + step])) code += step;
    const float c = c_tq6[code];
    const float num = tq_wsum(x * c), den = tq_wsum(c * c);
    unsigned u = code << (6 * (lane & 3));
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) u |= __shfl_xor_sync(0xffffffffu, u, o);
    packed = u;
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
