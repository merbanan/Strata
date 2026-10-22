// C02 experiment: independent routing and Q8 work in one grid. Compile with
// fast math to retain the router's contract. Q8 arithmetic explicitly retains
// subnormals and precise division. Compare every output byte to the engine.
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "router-row.cuh"

struct Q8 { half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q8) == 36);
#ifndef ROUTER_Q8_WARPS
#define ROUTER_Q8_WARPS 8
#endif
constexpr int fused_warps = ROUTER_Q8_WARPS;
__device__ __forceinline__ float precise_add(float a, float b) {
    float result; asm("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b)); return result;
}
__device__ __forceinline__ float precise_max(float a, float b) {
    float result; asm("max.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b)); return result;
}
__device__ __forceinline__ float precise_div(float a, float b) {
    float result; asm("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b)); return result;
}
__global__ __launch_bounds__(32 * fused_warps, 1) void fused(const float* logits, int32_t* ids, float* weights,
                                              const float* x, Q8* y, int rows, int cols) {
    const int route_blocks = (rows + fused_warps - 1) / fused_warps;
    if ((int) blockIdx.x < route_blocks) {
        router_probe::row(logits, ids, weights, rows, blockIdx.x * fused_warps + threadIdx.y);
        return;
    }
    const int i = ((int) blockIdx.x - route_blocks) * (32 * fused_warps) + threadIdx.y * 32 + threadIdx.x;
    if (i >= rows * cols) return;
    const float xi = x[i];
    float amax = __uint_as_float(__float_as_uint(xi) & 0x7fffffff), sum = xi;
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) {
        amax = precise_max(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
        sum = precise_add(sum, __shfl_xor_sync(0xffffffffu, sum, mask));
    }
    const float d = strata::kernels::q8_1_finite(precise_div(amax, 127.0f));
    float q = roundf(precise_div(xi, d));
    q = q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q;
    y[i / 32].qs[i % 32] = (__float_as_uint(amax) & 0x7fffffff) == 0 ? 0 : (int8_t) q;
    if (i % 32 == 0) y[i / 32].ds = strata::kernels::q8_1_ds(d, sum);
}
static void ck(cudaError_t error) {
    if (error != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(error)); std::exit(1); }
}
static void same(const void* a, const void* b, size_t size, const char* name, int rows, int cols, int seed) {
    const auto* x = (const unsigned char*) a;
    const auto* y = (const unsigned char*) b;
    for (size_t i = 0; i < size; ++i) if (x[i] != y[i]) {
        std::fprintf(stderr, "FAIL %s rows=%d cols=%d seed=%d byte=%zu reference=%02x fused=%02x\n",
                     name, rows, cols, seed, i, x[i], y[i]); std::exit(2);
    }
}
int main(int argc, char** argv) {
    bool timing = false, production = false, audit_routes = false, compare_production = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--time") == 0) timing = true;
        else if (std::strcmp(argv[i], "--production") == 0) production = true;
        else if (std::strcmp(argv[i], "--audit-route-differences") == 0) audit_routes = true;
        else if (std::strcmp(argv[i], "--compare-production-fusion") == 0) compare_production = true;
        else { std::fprintf(stderr, "unknown option: %s\n", argv[i]); return 1; }
    }
    const char* flag = std::getenv("STRATA_ROUTE_Q8_FUSE");
    const bool enabled = flag && std::strcmp(flag, "1") == 0;
    if (compare_production && (!timing || !enabled || production)) {
        std::fprintf(stderr, "--compare-production-fusion requires --time, STRATA_ROUTE_Q8_FUSE=1 and standalone mode\n");
        return 1;
    }
#if !defined(ROUTER_Q8_PRODUCTION_API)
    if (production || compare_production) {
        std::fprintf(stderr, "production fusion API is not in this source tree; apply the archived proposal and rebuild first\n");
        return 1;
    }
#endif
    cudaStream_t stream; ck(cudaStreamCreate(&stream));
    constexpr int max_rows = 31, max_cols = 5120;
    float *logits, *x, *weights[2]; int32_t* ids[2]; Q8* q[2];
    ck(cudaMalloc(&logits, max_rows * 512 * sizeof(float)));
    ck(cudaMalloc(&x, max_rows * max_cols * sizeof(float)));
    for (int arm = 0; arm < 2; ++arm) {
        ck(cudaMalloc(&weights[arm], (max_rows * 10 + 16) * sizeof(float)));
        ck(cudaMalloc(&ids[arm], (max_rows * 10 + 16) * sizeof(int32_t)));
        ck(cudaMalloc(&q[arm], (max_rows * max_cols / 32 + 16) * sizeof(Q8)));
    }
    std::mt19937 rng(0x2060);
    int cases = 0, different_id_cases = 0, different_weight_cases = 0;
    double maximum_weight_diff = 0.0;
    int nonfinite_weight_differences = 0;
    for (int rows : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 31}) for (int cols : {32, 2560, 5120}) {
        auto route_only = [&](int arm = 0) {
            if (rows == 1) strata::kernels::native_router_top10(logits, ids[arm], weights[arm], stream);
            else strata::kernels::native_router_top10_multi(logits, ids[arm], weights[arm], rows, stream);
        };
        auto launch = [&](int arm) {
            if (arm == 0) {
                route_only();
                strata::kernels::quantize_q8_1_rows(x, rows, cols, q[arm], stream);
            } else if (production) {
#if defined(ROUTER_Q8_PRODUCTION_API)
                const bool launched = strata::kernels::native_router_top10_q8_multi(
                    logits, ids[arm], weights[arm], x, q[arm], rows, cols, stream);
                if (launched != enabled) { std::fprintf(stderr, "incorrect opt-in dispatch\n"); std::exit(1); }
                if (!launched) {
                    route_only(arm);
                    strata::kernels::quantize_q8_1_rows(x, rows, cols, q[arm], stream);
                }
#else
                std::abort(); // rejected by option validation before allocating
#endif
            } else fused<<<(rows + fused_warps - 1) / fused_warps +
                           (rows * cols + 32 * fused_warps - 1) / (32 * fused_warps),
                           dim3(32, fused_warps), 0, stream>>>
                (logits, ids[arm], weights[arm], x, q[arm], rows, cols);
            ck(cudaGetLastError());
        };
        cudaGraph_t graph[2]; cudaGraphExec_t exec[2];
        for (int arm = 0; arm < 2; ++arm) {
            ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            launch(arm);
            ck(cudaStreamEndCapture(stream, &graph[arm]));
            ck(cudaGraphInstantiate(&exec[arm], graph[arm], nullptr, nullptr, 0));
        }
        for (int seed = 0; seed < 16; ++seed) {
            std::vector<float> hl(rows * 512), hx(rows * cols);
            for (size_t i = 0; i < hl.size(); ++i) {
                hl[i] = ((int) (rng() % 20001) - 10000) * 0.001f;
                if (seed == 1) hl[i] = 0; // ties select lower expert index
                if (seed == 2) hl[i] = i % 33 == 0 ? 8.0f : -8.0f;
                if (seed == 3) hl[i] *= 100.0f;
                if (seed == 4) hl[i] = std::ldexp(hl[i], -140);
                if (seed == 8 && i % 29 == 0) hl[i] = INFINITY;
                if (seed == 9 && i % 17 == 0) hl[i] = NAN;
                // Finite raw-logit ordering need not match rounded-softmax ties.
                if (seed == 12) hl[i] = i % 512 == 511 ? 0.0f : -1000.0f + (i % 512) * 0.01f;
                if (seed == 13) hl[i] = i % 512 < 10 ? 0.01f : i % 512 == 10 ? std::nextafter(0.01f, 1.0f) : -2.0f;
                if (seed == 14) hl[i] = -INFINITY;
                if (seed == 15 && i % 512 == 0) hl[i] = -INFINITY;
            }
            for (size_t i = 0; i < hx.size(); ++i) {
                const float v = ((int) (rng() % 20001) - 10000) * 0.001f;
                hx[i] = v;
                if (seed == 1) hx[i] = i % 2 ? -0.0f : 0.0f;
                if (seed == 2) hx[i] = std::ldexp(v, -140);
                if (seed == 3) hx[i] = std::ldexp(v, (int) ((i / 32) % 230) - 115);
                if (seed == 4) hx[i] = (i % 32 == 0) ? 127.0f : (i % 31) + 0.5f;
                if (seed == 5) hx[i] = std::nextafter((i % 32 == 0) ? 127.0f : (i % 31) + 0.5f, 0.0f);
                if (seed == 6) hx[i] = v * 1e7f;
                if (seed == 7) hx[i] = v * 1e-37f;
                if (seed == 8 && i % 29 == 0) hx[i] = INFINITY;
                if (seed == 9 && i % 17 == 0) hx[i] = NAN;
                if (seed == 10) hx[i] = std::nextafter((i % 32 == 0) ? 127.0f : (i % 31) + 0.5f, INFINITY);
                if (seed == 11) hx[i] = (i % 2 ? -1 : 1) * (65504.0f + i % 32);
            }
            ck(cudaMemcpyAsync(logits, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice, stream));
            ck(cudaMemcpyAsync(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice, stream));
            const size_t ib = (rows * 10 + 16) * 4, qb = (rows * cols / 32 + 16) * sizeof(Q8);
            std::vector<unsigned char> hi[2], hw[2], hq[2];
            for (int arm = 0; arm < 2; ++arm) {
                ck(cudaMemsetAsync(ids[arm], 0xa5, ib, stream));
                ck(cudaMemsetAsync(weights[arm], 0xa5, ib, stream));
                ck(cudaMemsetAsync(q[arm], 0xa5, qb, stream));
                if (seed & 1) ck(cudaGraphLaunch(exec[arm], stream)); else launch(arm);
                hi[arm].resize(ib); hw[arm].resize(ib); hq[arm].resize(qb);
                ck(cudaMemcpyAsync(hi[arm].data(), ids[arm], ib, cudaMemcpyDeviceToHost, stream));
                ck(cudaMemcpyAsync(hw[arm].data(), weights[arm], ib, cudaMemcpyDeviceToHost, stream));
                ck(cudaMemcpyAsync(hq[arm].data(), q[arm], qb, cudaMemcpyDeviceToHost, stream));
            }
            ck(cudaStreamSynchronize(stream));
            // Check the sentinel independently; equal overruns in both arms
            // must not count as an intact guard.
            for (int arm = 0; arm < 2; ++arm) {
                for (size_t i = rows * 10 * 4; i < ib; ++i)
                    if (hi[arm][i] != 0xa5 || hw[arm][i] != 0xa5) {
                        std::fprintf(stderr, "FAIL route guard arm=%d rows=%d cols=%d seed=%d\n", arm, rows, cols, seed);
                        return 2;
                    }
                for (size_t i = rows * cols / 32 * sizeof(Q8); i < qb; ++i)
                    if (hq[arm][i] != 0xa5) {
                        std::fprintf(stderr, "FAIL Q8 guard arm=%d rows=%d cols=%d seed=%d\n", arm, rows, cols, seed);
                        return 2;
                    }
            }
            if (audit_routes) {
                int id_differences = 0, weight_differences = 0, nonfinite = 0;
                double maximum = 0.0;
                for (int i = 0; i < rows * 10; ++i) {
                    id_differences += std::memcmp(hi[0].data() + i * 4, hi[1].data() + i * 4, 4) != 0;
                    if (std::memcmp(hw[0].data() + i * 4, hw[1].data() + i * 4, 4) != 0) {
                        ++weight_differences;
                        float a, b;
                        std::memcpy(&a, hw[0].data() + i * 4, 4);
                        std::memcpy(&b, hw[1].data() + i * 4, 4);
                        if (std::isfinite(a) && std::isfinite(b)) maximum = std::max(maximum, std::abs((double) a - b));
                        else ++nonfinite;
                    }
                }
                different_id_cases += id_differences != 0;
                different_weight_cases += weight_differences != 0;
                maximum_weight_diff = std::max(maximum_weight_diff, maximum);
                nonfinite_weight_differences += nonfinite;
                if (id_differences || weight_differences)
                    std::printf("route_diff rows=%d cols=%d seed=%d ids=%d weights=%d max_abs=%.9g nonfinite=%d\n",
                                rows, cols, seed, id_differences, weight_differences, maximum, nonfinite);
            } else {
                same(hi[0].data(), hi[1].data(), ib, "ids", rows, cols, seed);
                same(hw[0].data(), hw[1].data(), ib, "weights", rows, cols, seed);
            }
            same(hq[0].data(), hq[1].data(), qb, "q8", rows, cols, seed);
            ++cases;
        }
        if (timing && cols == 2560) {
            // Timing uses ordinary finite activations, not the last NaN case.
            std::vector<float> hl(rows * 512), hx(rows * cols);
            for (auto& value : hl) value = ((int) (rng() % 20001) - 10000) * 0.001f;
            for (auto& value : hx) value = ((int) (rng() % 20001) - 10000) * 0.001f;
            ck(cudaMemcpyAsync(logits, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice, stream));
            ck(cudaMemcpyAsync(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice, stream));
            ck(cudaStreamSynchronize(stream));
            const int timing_arms = compare_production ? 4 : 3;
            cudaGraph_t batch[4]; cudaGraphExec_t batch_exec[4];
            for (int arm = 0; arm < timing_arms; ++arm) {
                ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
                for (int i = 0; i < 128; ++i) {
                    if (arm == 2) route_only();
                    else if (arm == 3) {
#if defined(ROUTER_Q8_PRODUCTION_API)
                        if (!strata::kernels::native_router_top10_q8_multi(logits, ids[1], weights[1], x, q[1],
                                                                         rows, cols, stream)) {
                            std::fprintf(stderr, "production fusion comparison did not launch\n");
                            return 1;
                        }
#else
                        std::abort(); // rejected by option validation
#endif
                    } else launch(arm);
                }
                ck(cudaStreamEndCapture(stream, &batch[arm]));
                ck(cudaGraphInstantiate(&batch_exec[arm], batch[arm], nullptr, nullptr, 0));
            }
            cudaEvent_t start, end; ck(cudaEventCreate(&start)); ck(cudaEventCreate(&end));
            // Arm 2 isolates the reference router completion. The fused launch
            // must finish Q8 work too before a following doorbell can execute.
            const std::vector<int> order = compare_production ? std::vector<int>{0, 1, 2, 3, 3, 2, 1, 0}
                                                              : std::vector<int>{0, 1, 2, 2, 1, 0};
            for (int arm : order) {
                for (int i = 0; i < 4; ++i) ck(cudaGraphLaunch(batch_exec[arm], stream));
                ck(cudaEventRecord(start, stream));
                for (int i = 0; i < 16; ++i) ck(cudaGraphLaunch(batch_exec[arm], stream));
                ck(cudaEventRecord(end, stream)); ck(cudaEventSynchronize(end));
                float ms; ck(cudaEventElapsedTime(&ms, start, end));
                std::printf("timing rows=%d arm=%d graph_us=%.4f\n", rows, arm, ms * 1000 / (128 * 16));
            }
            ck(cudaEventDestroy(start)); ck(cudaEventDestroy(end));
            for (int arm = 0; arm < timing_arms; ++arm) { ck(cudaGraphExecDestroy(batch_exec[arm])); ck(cudaGraphDestroy(batch[arm])); }
        }
        for (int arm = 0; arm < 2; ++arm) { ck(cudaGraphExecDestroy(exec[arm])); ck(cudaGraphDestroy(graph[arm])); }
    }
    for (int arm = 0; arm < 2; ++arm) { ck(cudaFree(ids[arm])); ck(cudaFree(weights[arm])); ck(cudaFree(q[arm])); }
    ck(cudaFree(logits)); ck(cudaFree(x)); ck(cudaStreamDestroy(stream));
    if (audit_routes) {
        std::printf("AUDIT cases=%d different_id_cases=%d different_weight_cases=%d max_abs=%.9g nonfinite=%d Q8_and_guards_exact=1\n",
                    cases, different_id_cases, different_weight_cases, maximum_weight_diff, nonfinite_weight_differences);
        return different_id_cases || different_weight_cases ? 2 : 0;
    }
    std::printf("PASS %d cases, exact routes/weights/Q8 bytes, guards and graph replay\n", cases);
}
