// Build (server): nvcc -std=c++20 -ccbin g++-12 -arch=sm_70 -Iinclude bench/micro/qsa_sel_bench.cu build/libstrata_kernels.a build/libstrata_kernels_cpu.a -lcuda
// STRATA_QSA_TOPK_MULTI=0 / 1 for the two top-k paths; the ids checksum must match.
// QSA decode selection cost split: qsa_block_scores vs qsa_block_topk for one query at 30K / 60K / 120K cells
// (max_blocks for a 131072 context), random pooled keys; 200 launches each, events.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include "strata/kernels/qsa_select.hpp"
using namespace strata::kernels;
int main() {
    const QsaShapes s = qsa_real_shapes();
    const int64_t max_blocks = 131072 / 4 + 2;
    const int64_t cap = qsa_selection_width(kTopkMaxCells, s);
    std::vector<float> h((size_t) max_blocks * 128);
    std::mt19937 rng(1); std::normal_distribution<float> nd;
    for (auto& v : h) v = std::round(nd(rng) * 4.0f) / 4.0f;   // coarse: equal scores (ties) occur
    float *pooled, *dead, *q, *scores; int32_t *steps, *ids;
    cudaMalloc(&pooled, h.size() * 4); cudaMemcpy(pooled, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dead, 4 * 128 * 4 * 4); cudaMemset(dead, 0, 4 * 128 * 4 * 4);
    cudaMalloc(&q, 4 * 128 * 4); cudaMemcpy(q, h.data(), 4 * 128 * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&scores, max_blocks * 4); cudaMalloc(&steps, 16); cudaMalloc(&ids, cap * 4);
    cudaStream_t st; cudaStreamCreate(&st);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int64_t cells : {1500, 3000, 30000, 60000, 120000}) {
        const int32_t hs[4] = {(int32_t) cells - 1, (int32_t) cells, (int32_t) (cells / 4), (int32_t) qsa_selection_width(cells, s)};
        cudaMemcpy(steps, hs, 16, cudaMemcpyHostToDevice);
        float t[2];
        for (int k = 0; k < 2; ++k) {
            for (int i = 0; i < 10; ++i) {
                if (k == 0) qsa_block_scores(pooled, dead, q, steps, 1, max_blocks, s, scores, st);
                else qsa_block_topk(scores, steps, 1, max_blocks, cap, s, ids, st);
            }
            // as the engine runs it: 200 calls captured in one graph, so the host's launches do not pace the GPU
            cudaGraph_t gr; cudaGraphExec_t ge;
            cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
            for (int i = 0; i < 200; ++i) {
                if (k == 0) qsa_block_scores(pooled, dead, q, steps, 1, max_blocks, s, scores, st);
                else qsa_block_topk(scores, steps, 1, max_blocks, cap, s, ids, st);
            }
            cudaStreamEndCapture(st, &gr);
            cudaGraphInstantiate(&ge, gr, nullptr, nullptr, 0);
            cudaGraphLaunch(ge, st); cudaStreamSynchronize(st);
            cudaEventRecord(a, st);
            cudaGraphLaunch(ge, st);
            cudaEventRecord(b, st); cudaEventSynchronize(b);
            cudaGraphExecDestroy(ge); cudaGraphDestroy(gr);
            cudaEventElapsedTime(&t[k], a, b); t[k] = t[k] * 1000 / 200;
        }
        std::vector<int32_t> hid((size_t) cap);
        cudaMemcpy(hid.data(), ids, cap * 4, cudaMemcpyDeviceToHost);
        unsigned long long ck = 0;
        for (int64_t i = 0; i < hs[3]; ++i) ck = ck * 1000003ull + (unsigned) hid[(size_t) i];
        std::printf("%6lld cells: scores %7.1f us  topk %7.1f us  ids %016llx\n", (long long) cells, t[0], t[1], ck);
    }
    std::printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
