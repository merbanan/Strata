// Check the opt-in PF_DM_GROUP's expert-order sum and reused group scratch.
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
constexpr int T = 4, K = 10, E = 35, N = 2560, R = T * K;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
template<typename V> V* upload(const std::vector<V>& h) {
    V* p = nullptr;
    ck(cudaMalloc(&p, h.size() * sizeof(V)), "malloc");
    ck(cudaMemcpy(p, h.data(), h.size() * sizeof(V), cudaMemcpyHostToDevice), "upload");
    return p;
}
template<typename V> V* alloc(size_t n) {
    V* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(V)), "malloc");
    return p;
}
bool bits_equal(float a, float b) { return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b); }
} // namespace

int main() {
    std::vector<int32_t> choice(T * K), slot(T * K), count(E), bounds(E + 1), src(R);
    std::vector<float> w(T * K), rows((size_t) R * N), shared((size_t) T * N, 0.0f), sg(T, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) {
            const int i = t * K + k, e = (3 * t + 11 * k) % E;
            choice[i] = e;
            ++count[e];
            w[i] = 0.03f + 0.011f * (float) ((t + 3 * k) % 9);
        }
    for (int e = 0; e < E; ++e) bounds[e + 1] = bounds[e] + count[e];
    std::vector<int32_t> next = bounds;
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) {
            const int i = t * K + k, r = next[choice[i]]++;
            slot[i] = r;
            src[r] = t;
        }
    for (int r = 0; r < R; ++r)
        for (int d = 0; d < N; ++d) {
            float v = std::sin((float) ((r + 1) * (d + 3)) * 0.013f) * (float) (1 + r % 7);
            if (d == 0) v = (r % 3 == 0 ? 1.0e8f : r % 3 == 1 ? -1.0e8f : 1.0f);
            rows[(size_t) r * N + d] = v;
        }

    const auto* dslot = upload(slot); const auto* dsrc = upload(src);
    const auto* dbounds = upload(bounds); const auto* dw = upload(w);
    const auto* drows = upload(rows); const auto* dshared = upload(shared); const auto* dsg = upload(sg);
    auto* drow_w = alloc<float>(R); auto* dacc = alloc<float>((size_t) T * N);
    auto* dbase = alloc<float>((size_t) T * N); auto* dgroup = alloc<float>((size_t) T * N);
    int max_group_rows = 0;
    for (int e = 0; e < E; e += 3)
        max_group_rows = std::max(max_group_rows, bounds[std::min(E, e + 3)] - bounds[e]);
    auto* dscratch = alloc<float>((size_t) max_group_rows * N);
    ck(cudaMemset(dacc, 0, (size_t) T * N * sizeof(float)), "clear accumulator");
    strata::prefill::moe_row_weights(dslot, dw, drow_w, R, nullptr);
    strata::prefill::moe_combine(drows, dslot, dw, dshared, dsg, dbase, T, nullptr);
    for (int e = 0; e < E; e += 3) {
        const int e1 = std::min(E, e + 3), ra = bounds[e], nr = bounds[e1] - ra;
        if (nr) ck(cudaMemcpyAsync(dscratch, drows + (size_t) ra * N, (size_t) nr * N * sizeof(float),
                                  cudaMemcpyDeviceToDevice), "reuse group scratch");
        strata::prefill::moe_group_accum(dacc, dscratch, dsrc, drow_w, dbounds + e, e1 - e, ra, nullptr);
    }
    strata::prefill::moe_combine_acc(dacc, dshared, dsg, dgroup, T, nullptr);
    ck(cudaDeviceSynchronize(), "group completion");
    std::vector<float> base((size_t) T * N), group((size_t) T * N);
    ck(cudaMemcpy(base.data(), dbase, base.size() * sizeof(float), cudaMemcpyDeviceToHost), "read base");
    ck(cudaMemcpy(group.data(), dgroup, group.size() * sizeof(float), cudaMemcpyDeviceToHost), "read group");
    size_t wrong_base = 0, wrong_group = 0, different = 0, different_typical = 0;
    float max_abs = 0.0f, max_abs_typical = 0.0f;
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < N; ++d) {
            float bk = 0.0f, ge = 0.0f;
            for (int k = 0; k < K; ++k) {
                const int i = t * K + k;
                bk = std::fma(w[i], rows[(size_t) slot[i] * N + d], bk);
            }
            for (int e = 0; e < E; ++e)
                for (int k = 0; k < K; ++k) {
                    const int i = t * K + k;
                    if (choice[i] == e) ge = std::fma(w[i], rows[(size_t) slot[i] * N + d], ge);
                }
            const size_t i = (size_t) t * N + d;
            wrong_base += !bits_equal(base[i], bk);
            wrong_group += !bits_equal(group[i], ge);
            different += !bits_equal(base[i], group[i]);
            max_abs = std::max(max_abs, std::abs(base[i] - group[i]));
            if (d > 0) {
                different_typical += !bits_equal(base[i], group[i]);
                max_abs_typical = std::max(max_abs_typical, std::abs(base[i] - group[i]));
            }
        }
    const int empty = (int) std::count(count.begin(), count.end(), 0);
    std::printf("PF_DM_GROUP: base_reference_errors=%zu group_reference_errors=%zu different_outputs=%zu/%zu "
                "ordinary_different=%zu/%zu max_abs=%.9g ordinary_max_abs=%.9g empty_experts=%d\n",
                wrong_base, wrong_group, different, base.size(), different_typical, (size_t) T * (N - 1),
                max_abs, max_abs_typical, empty);
    cudaFree((void*) dslot); cudaFree((void*) dsrc); cudaFree((void*) dbounds); cudaFree((void*) dw);
    cudaFree((void*) drows); cudaFree((void*) dshared); cudaFree((void*) dsg);
    cudaFree(drow_w); cudaFree(dacc); cudaFree(dbase); cudaFree(dgroup); cudaFree(dscratch);
    return wrong_base || wrong_group || !different || !empty ? 1 : 0;
}
