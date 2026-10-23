// src/kernels/qsa_prompt_attn_parity.cpp - perf-review D-1: the tensor-core prompt attention (qsa_prompt_attn.hpp)
// against the FP32 kernel it replaces (`qsa_decode_attn_batch`) and an FP64 host reference (GPU, synthetic, no model).
//
// Int8, FP16 and Q4_0 pools with random codes, scales and queries; selections shaped like the prompt path's (the 2,051
// widest, a recent window plus older cells that drift slowly from one query to the next, so neighbours share most
// of them as they do in a real prompt; short contexts take every cell). Checks:
//   1. against FP64, the new kernel's error is no larger than a small multiple of the old kernel's (both FP32 math);
//   2. the new and old outputs agree to a relative 1e-4 of the output scale;
// then times both over a prompt chunk (the old one in batches of 32, as prefill.cpp calls it).
// HIP builds (S6): the same checks for the RDNA4 matrix-core kernel (STRATA_HIP_WMMA), skipped (77) off gfx11 / gfx12.
//   4. (HIP, with STRATA_PA_FAST=1) the gated epilogue (qsa_prompt_attn_batch_gated) writes the same FP16 bits as the
//      attention followed by the prompt path's gate_attn, for random gates.
// Hybrid K8V4 coverage includes the existing Q4-to-half rounding boundary.
// QA_GRAPH=1 replays with changed queries and poisoned output; QA_EMPTY=1 includes
// a zero-width selection. QA_OUTPUT writes output floats for cross-process A/B.
// Usage: qsa_prompt_attn_parity [context=32768] [queries=2048] [reps=5] [format=0|1|3; default all]
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/kv_q4.hpp"
#if defined(__HIP_PLATFORM_AMD__)
#include "strata/prefill/kernels.hpp"
#endif

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T) + 64), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
float h2f(uint16_t b) { __half h; *reinterpret_cast<uint16_t*>(&h) = b; return __half2float(h); }
uint16_t f2h(float f) { __half h = __float2half(f); return *reinterpret_cast<uint16_t*>(&h); }

int run(int fmt, int64_t ctx, int64_t nq, int reps) {   // fmt 1 int8, 0 fp16, 2 q4_0, 3 k8v4
#if !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
    if (fmt == 2) {   // mode 4 (Q4_0 KV) runs on sm_80 and newer only: below that the dispatcher keeps the old kernel
        int dev = 0, major = 0;
        ck(cudaGetDevice(&dev), "device");
        ck(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev), "cc");
        if (major < 8) {
            std::printf("SKIP q4_0 ctx %lld: the tensor-core kernel takes Q4_0 KV on sm_80+ only (this device: sm_%d)\n",
                        (long long) ctx, major);
            return 0;
        }
    }
#endif
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, NH = s.n_head, PS = s.page_size;
    const int64_t pages = (ctx + PS - 1) / PS, rows = pages * NKV * PS;
    std::mt19937 rng(1234 + fmt);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.03f);
    // pools; page table a shuffled permutation (the readers must follow it)
    std::vector<int8_t> kq, vq;
    std::vector<uint16_t> ks, vs, kh, vh;
    std::vector<uint8_t> k4, v4;   // q4_0: 8 blocks of {fp16 d, 16 code bytes} per row (kv_q4.hpp)
    constexpr int64_t B4 = 18, ROW4 = 8 * B4;
    if (fmt == 2 || fmt == 3) {
        std::uniform_int_distribution<int> byte(0, 255);
        // the scales as a q4_0 pool holds them: signed (ggml's d = max / -8) and spread over two decades, so a chunk's
        // cells mix signs and magnitudes (a positive-only bound on them overflowed FP16 on real K/V)
        std::uniform_real_distribution<float> lg(std::log(0.01f), std::log(1.0f));
        auto sc4 = [&](std::mt19937& r) { return std::exp(lg(r)) * (byte(r) & 1 ? -1.0f : 1.0f); };
        k4.resize(rows * ROW4); v4.resize(rows * ROW4);
        for (auto* pool : {&k4, &v4})
            for (int64_t b = 0; b < rows * 8; ++b) {
                const uint16_t d = f2h(sc4(rng));
                std::memcpy(pool->data() + b * B4, &d, 2);
                for (int j = 0; j < 16; ++j) (*pool)[b * B4 + 2 + j] = (uint8_t) byte(rng);
            }
        if (fmt == 3) {
            kq.resize(rows * HD); ks.resize(rows * 4);
            for (auto& x : kq) x = (int8_t) code(rng);
            for (auto& x : ks) x = f2h(sc(rng));
        }
    } else if (fmt == 1) {
        kq.resize(rows * HD); vq.resize(rows * HD); ks.resize(rows * 4); vs.resize(rows * 4);
        for (auto& x : kq) x = (int8_t) code(rng);
        for (auto& x : vq) x = (int8_t) code(rng);
        for (auto& x : ks) x = f2h(sc(rng));
        for (auto& x : vs) x = f2h(sc(rng));
    } else {
        kh.resize(rows * HD); vh.resize(rows * HD);
        for (auto& x : kh) x = f2h(nd(rng) * 1.5f);
        for (auto& x : vh) x = f2h(nd(rng));
    }
    std::vector<int32_t> table(pages);
    for (int64_t i = 0; i < pages; ++i) table[i] = (int32_t) i;
    std::shuffle(table.begin(), table.end(), rng);
    // queries at positions ctx - nq .. ctx - 1
    const int64_t cap = k::qsa_selection_width(ctx, s);
    std::vector<int32_t> ids((size_t) (nq * cap), 0), steps((size_t) (nq * k::kStepCount), 0);
    std::vector<float> q((size_t) (nq * NH * HD));
    for (auto& x : q) x = nd(rng) * 2.0f;
    std::vector<int32_t> old_cells;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t pos = ctx - nq + i, nkv = pos + 1;
        const int64_t w = k::qsa_selection_width(nkv, s);
        int32_t* sel = ids.data() + i * cap;
        steps[i * k::kStepCount + k::kStepWidth] = (int32_t) w;
        if (w == nkv) {
            for (int64_t c = 0; c < w; ++c) sel[c] = (int32_t) c;
            continue;
        }
        const int64_t recent = 512, older = w - recent;
        if ((int64_t) old_cells.size() != older) {   // the older cells: a set that drifts ~3% per query
            std::vector<int32_t> all((size_t) (nkv - recent));
            for (int64_t c = 0; c < nkv - recent; ++c) all[c] = (int32_t) c;
            std::shuffle(all.begin(), all.end(), rng);
            old_cells.assign(all.begin(), all.begin() + older);
        } else {
            std::uniform_int_distribution<int64_t> pick(0, older - 1), any(0, nkv - recent - 1);
            for (int r = 0; r < older / 32; ++r) {
                const int32_t c = (int32_t) any(rng);
                if (std::find(old_cells.begin(), old_cells.end(), c) == old_cells.end()) old_cells[pick(rng)] = c;
            }
        }
        std::vector<int32_t> v(old_cells);
        for (int64_t c = nkv - recent; c < nkv; ++c) v.push_back((int32_t) c);
        std::sort(v.begin(), v.end());
        std::copy(v.begin(), v.end(), sel);
    }
    // QA_SAME=1 (S23 probe): every query reads the same first-w cells - the best case of any K/V sharing between queries
    // (all loads hit the caches); QA_SAME=2: query i reads a window of w cells at i / 4 * 64 (neighbours share ~90%)
    if (const char* sm = std::getenv("QA_SAME")) {
        const int mode = std::atoi(sm);
        for (int64_t i = 0; i < nq; ++i) {
            const int64_t w = steps[i * k::kStepCount + k::kStepWidth], nkv = ctx - nq + i + 1;
            if (w == nkv) continue;
            if (mode == 3 || mode == 4) {   // 3: w random cells per query (no overlap beyond chance); 4: a 4-query group shares
                // one random set of w - 512 cells (the union of 4 queries = 1x), the last 512 cells recent
                std::mt19937 r3((unsigned) (mode == 3 ? i : i / 4) * 2654435761u + 7);
                std::vector<int32_t> all((size_t) (nkv - 512));
                for (int64_t c = 0; c < nkv - 512; ++c) all[c] = (int32_t) c;
                std::shuffle(all.begin(), all.end(), r3);
                std::vector<int32_t> v(all.begin(), all.begin() + (w - 512));
                for (int64_t c = nkv - 512; c < nkv; ++c) v.push_back((int32_t) c);
                std::sort(v.begin(), v.end());
                std::copy(v.begin(), v.end(), ids.data() + i * cap);
                continue;
            }
            const int64_t base = mode == 1 ? 0 : std::min<int64_t>(i / 4 * 64, nkv - w);
            for (int64_t c = 0; c < w; ++c) ids[i * cap + c] = (int32_t) (base + c);
        }
    }
    // QA_DUMP=<file> QA_LAYER=n (S23 probe): the engine's own selections (STRATA_QSA_DUMP records {layer, pos0, T, cap} +
    // T*cap cells): the last record of that layer, whose T must be nq and pos0 + T ctx
    if (const char* df = std::getenv("QA_DUMP")) {
        const int want = std::getenv("QA_LAYER") ? std::atoi(std::getenv("QA_LAYER")) : 3;
        std::FILE* f = std::fopen(df, "rb");
        if (!f) { std::fprintf(stderr, "QA_DUMP: cannot open\n"); std::exit(2); }
        int32_t h[4];
        long at = -1;
        while (std::fread(h, 4, 4, f) == 4) {
            const long data = std::ftell(f);
            if (h[0] == want && h[2] == nq && h[1] + h[2] == ctx && h[3] == cap) at = data;
            std::fseek(f, (long) h[2] * h[3] * 4, SEEK_CUR);
        }
        if (at < 0) { std::fprintf(stderr, "QA_DUMP: no record for layer %d, T %lld, ctx %lld\n", want, (long long) nq, (long long) ctx); std::exit(2); }
        std::fseek(f, at, SEEK_SET);
        if (std::fread(ids.data(), 4, ids.size(), f) != ids.size()) std::exit(2);
        std::fclose(f);
        for (int grp : {2, 4, 8, 16}) {   // the union of `grp` neighbouring queries' cells against grp x the width
            double su = 0, sw = 0;
            for (int64_t i0 = 0; i0 + grp <= nq; i0 += grp * 16) {
                std::vector<int32_t> u;
                for (int g2 = 0; g2 < grp; ++g2) {
                    const int64_t w = steps[(i0 + g2) * k::kStepCount + k::kStepWidth];
                    u.insert(u.end(), ids.begin() + (i0 + g2) * cap, ids.begin() + (i0 + g2) * cap + w);
                    sw += (double) w;
                }
                std::sort(u.begin(), u.end());
                su += (double) (std::unique(u.begin(), u.end()) - u.begin());
            }
            std::printf("QA_DUMP overlap: %d-query groups read %.2fx the cells of one query (%.1f%% of %d x)\n", grp,
                        su * grp / sw, 100.0 * su / sw, grp);
        }
    }
    if (std::getenv("QA_EMPTY")) steps[k::kStepWidth] = 0;
    k::QsaAttnPools pl;
    if (fmt == 3) { pl.k_q = up(kq); pl.k_scale = up(ks); pl.v_q4 = up(v4); }
    else if (fmt == 2) { pl.k_q4 = up(k4); pl.v_q4 = up(v4); }
    else if (fmt == 1) { pl.k_q = up(kq); pl.v_q = up(vq); pl.k_scale = up(ks); pl.v_scale = up(vs); }
    else { pl.k_pool = up(kh); pl.v_pool = up(vh); }
    // a q4_0 value: block d / 32 of the row, element j = d % 32 in the low nibble of byte j (j < 16), else the high
    // nibble of byte j - 16, minus 8, times the block's scale
    auto q4v = [&](const std::vector<uint8_t>& pool, int64_t row, int64_t d) {
        const uint8_t* blk = pool.data() + row * ROW4 + (d / 32) * B4;
        uint16_t sc;
        std::memcpy(&sc, blk, 2);
        const int j = (int) (d % 32);
        const int code = j < 16 ? (blk[2 + j] & 0x0F) : (blk[2 + j - 16] >> 4);
        return (double) (code - 8) * h2f(sc);
    };
    pl.page_table = up(table);
    const int32_t* d_ids = up(ids);
    const int32_t* d_steps = up(steps);
    const float* d_q = up(q);
    float *d_old = nullptr, *d_new = nullptr, *scratch = nullptr;
    const int64_t batch = 32;
    ck(cudaMalloc(&d_old, nq * NH * HD * 4), "malloc");
    ck(cudaMalloc(&d_new, nq * NH * HD * 4), "malloc");
    ck(cudaMalloc(&scratch, batch * k::qsa_decode_attn_scratch_floats(cap, s) * 4), "malloc");
    auto old_run = [&]() {
        for (int64_t t0 = 0; t0 < nq; t0 += batch)
            k::qsa_decode_attn_batch(d_q + t0 * NH * HD, pl, d_ids + t0 * cap, d_steps + t0 * k::kStepCount, cap, s,
                                     scratch, d_old + t0 * NH * HD, std::min(batch, nq - t0), nullptr);
    };
    auto new_run = [&]() {
        if (!k::qsa_prompt_attn_batch(d_q, pl, d_ids, d_steps, cap, s, d_new, nq, nullptr)) {
            std::fprintf(stderr, "qsa_prompt_attn_batch refused the pools\n");
            std::exit(2);
        }
    };
    old_run();
    new_run();
    ck(cudaDeviceSynchronize(), "run");
    if (std::getenv("QA_GRAPH")) {
        cudaStream_t stream;
        ck(cudaStreamCreate(&stream), "graph stream");
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture begin");
        if (!k::qsa_prompt_attn_batch(d_q, pl, d_ids, d_steps, cap, s, d_new, nq, stream)) std::exit(2);
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        ck(cudaStreamEndCapture(stream, &graph), "capture end");
        ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
        for (int replay = 0; replay < 3; ++replay) {
            for (auto& x : q) x = -x * 0.9375f;
            ck(cudaMemcpy((void*) d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice), "changed queries");
            ck(cudaMemsetAsync(d_new, 0xff, nq * NH * HD * sizeof(float), stream), "poison output");
            ck(cudaGraphLaunch(exec, stream), "graph replay");
            ck(cudaStreamSynchronize(stream), "graph sync");
        }
        old_run();
        ck(cudaDeviceSynchronize(), "changed reference");
        ck(cudaGraphExecDestroy(exec), "destroy exec");
        ck(cudaGraphDestroy(graph), "destroy graph");
        ck(cudaStreamDestroy(stream), "destroy stream");
    }
    std::vector<float> o((size_t) (nq * NH * HD)), nw(o.size());
    ck(cudaMemcpy(o.data(), d_old, o.size() * 4, cudaMemcpyDeviceToHost), "down");
    ck(cudaMemcpy(nw.data(), d_new, nw.size() * 4, cudaMemcpyDeviceToHost), "down");
    {   // a hash of the new kernel's output bits (bitwise A/B of two builds / switches)
        uint64_t hh = 1469598103934665603ull;
        for (float x : nw) { uint32_t u; std::memcpy(&u, &x, 4); hh = (hh ^ u) * 1099511628211ull; }
        std::printf("NEWHASH %016llx\n", (unsigned long long) hh);
    }
    if (const char* path = std::getenv("QA_OUTPUT")) {
        std::FILE* f = std::fopen(path, "wb");
        if (!f || std::fwrite(nw.data(), sizeof(float), nw.size(), f) != nw.size()) std::exit(2);
        if (std::fclose(f)) std::exit(2);
    }
    bool ok_merge = true;
#if !defined(__HIP_PLATFORM_AMD__)
    if ((fmt == 0 || fmt == 3) && std::getenv("QA_WINDOW_MERGE")) {
        std::vector<int32_t> ids_a(ids.size(), 0), ids_b(ids.size(), 0);
        std::vector<int32_t> steps_a = steps, steps_b = steps;
        for (int64_t i = 0; i < nq; ++i) {
            const int w = steps[i * k::kStepCount + k::kStepWidth];
            const int cut = i % 3 == 0 ? 0 : i % 3 == 1 ? w : w / 2;
            steps_a[i * k::kStepCount + k::kStepWidth] = cut;
            steps_b[i * k::kStepCount + k::kStepWidth] = w - cut;
            std::copy_n(ids.data() + i * cap, cut, ids_a.data() + i * cap);
            std::copy_n(ids.data() + i * cap + cut, w - cut, ids_b.data() + i * cap);
        }
        const int32_t* da = up(ids_a); const int32_t* db = up(ids_b);
        const int32_t* sa = up(steps_a); const int32_t* sb = up(steps_b);
        float *oa = nullptr, *ob = nullptr, *la = nullptr, *lb = nullptr;
        ck(cudaMalloc(&oa, nw.size() * sizeof(float)), "slice A output");
        ck(cudaMalloc(&ob, nw.size() * sizeof(float)), "slice B output");
        ck(cudaMalloc(&la, (size_t) (nq * NH) * sizeof(float)), "slice A lse");
        ck(cudaMalloc(&lb, (size_t) (nq * NH) * sizeof(float)), "slice B lse");
        if (!k::qsa_prompt_attn_batch_lse(d_q, pl, da, sa, cap, s, oa, la, nq, nullptr) ||
            !k::qsa_prompt_attn_batch_lse(d_q, pl, db, sb, cap, s, ob, lb, nq, nullptr)) std::exit(2);
        ck(cudaDeviceSynchronize(), "slice attention");
        std::vector<float> ha(nw.size()), hb(nw.size()), hla((size_t) (nq * NH)), hlb(hla.size());
        ck(cudaMemcpy(ha.data(), oa, ha.size() * 4, cudaMemcpyDeviceToHost), "slice A down");
        ck(cudaMemcpy(hb.data(), ob, hb.size() * 4, cudaMemcpyDeviceToHost), "slice B down");
        ck(cudaMemcpy(hla.data(), la, hla.size() * 4, cudaMemcpyDeviceToHost), "slice A lse down");
        ck(cudaMemcpy(hlb.data(), lb, hlb.size() * 4, cudaMemcpyDeviceToHost), "slice B lse down");
        double worst = 0, magnitude = 0;
        for (int64_t i = 0; i < nq * NH; ++i) {
            const double a = hla[i], b = hlb[i], m = std::max(a, b);
            const double wa = std::isfinite(a) ? std::exp2(a - m) : 0.0;
            const double wb = std::isfinite(b) ? std::exp2(b - m) : 0.0;
            if (!(std::isfinite(a) || a == -INFINITY) || !(std::isfinite(b) || b == -INFINITY)) ok_merge = false;
            for (int64_t d = 0; d < HD; ++d) {
                const size_t at = (size_t) i * HD + d;
                const double merged = wa + wb > 0.0 ? (wa * ha[at] + wb * hb[at]) / (wa + wb) : 0.0;
                worst = std::max(worst, std::fabs(merged - nw[at]));
                magnitude = std::max(magnitude, (double) std::fabs(nw[at]));
            }
        }
        ok_merge = ok_merge && worst <= 3e-4 * std::max(magnitude, 1.0);
        std::printf("%s %s two-window LSE merge: max difference %.3g (output scale %.3g)\n",
                    ok_merge ? "PASS" : "FAIL", fmt == 3 ? "K8V4" : "FP16", worst, magnitude);
        cudaFree((void*) da); cudaFree((void*) db); cudaFree((void*) sa); cudaFree((void*) sb);
        cudaFree(oa); cudaFree(ob); cudaFree(la); cudaFree(lb);
    }
    if ((fmt == 0 || fmt == 3) && std::getenv("QA_BOUNDED_STAGE")) {
        const int requested = std::atoi(std::getenv("QA_BOUNDED_STAGE"));
        if (requested < 1) { std::fprintf(stderr, "QA_BOUNDED_STAGE needs a positive page count\n"); std::exit(2); }
        std::vector<uint8_t> wanted((size_t) pages, 0);
        for (int64_t qi = 0; qi < nq; ++qi)
            for (int j = 0; j < steps[qi * k::kStepCount + k::kStepWidth]; ++j)
                wanted[(size_t) (ids[qi * cap + j] / PS)] = 1;
        std::vector<int32_t> selected_pages;
        for (int64_t page = 0; page < pages; ++page)
            if (wanted[(size_t) page]) selected_pages.push_back((int32_t) page);
        const int slots = std::min<int64_t>(requested, std::max<size_t>(selected_pages.size(), 1));
        const size_t block_rows = (size_t) (NKV * PS);
        const size_t k_bytes = block_rows * HD * (fmt == 3 ? 1 : 2);
        const size_t scale_bytes = block_rows * 4 * 2;
        const size_t v_bytes = block_rows * (fmt == 3 ? ROW4 : HD * 2);
        auto alloc = [](size_t bytes) {
            void* p = nullptr;
            ck(cudaMalloc(&p, bytes), "bounded stage alloc");
            return p;
        };
        k::QsaAttnPools staged;
        void* stage_k = alloc((size_t) slots * k_bytes);
        void* stage_v = alloc((size_t) slots * v_bytes);
        void* stage_scale = fmt == 3 ? alloc((size_t) slots * scale_bytes) : nullptr;
        void *bounce_k = nullptr, *bounce_v = nullptr, *bounce_scale = nullptr;
        ck(cudaHostAlloc(&bounce_k, (size_t) slots * k_bytes, cudaHostAllocDefault), "bounded K bounce");
        ck(cudaHostAlloc(&bounce_v, (size_t) slots * v_bytes, cudaHostAllocDefault), "bounded V bounce");
        if (fmt == 3) ck(cudaHostAlloc(&bounce_scale, (size_t) slots * scale_bytes, cudaHostAllocDefault),
                         "bounded scale bounce");
        int32_t* stage_table = (int32_t*) alloc((size_t) pages * 4);
        staged.page_table = stage_table;
        if (fmt == 3) {
            staged.k_q = (int8_t*) stage_k;
            staged.k_scale = (uint16_t*) stage_scale;
            staged.v_q4 = (uint8_t*) stage_v;
        } else {
            staged.k_pool = (uint16_t*) stage_k;
            staged.v_pool = (uint16_t*) stage_v;
        }
        int32_t* stage_ids = (int32_t*) alloc(ids.size() * 4);
        int32_t* stage_steps = (int32_t*) alloc(steps.size() * 4);
        float* stage_out = (float*) alloc(nw.size() * 4);
        float* stage_lse = (float*) alloc((size_t) (nq * NH) * 4);
        std::vector<int32_t> map((size_t) pages, -1), part_ids(ids.size(), 0), part_steps = steps;
        std::vector<int64_t> covered((size_t) nq, 0);
        std::vector<float> part_out(nw.size()), part_lse((size_t) (nq * NH));
        std::vector<double> merged(nw.size(), 0.0), merged_lse(part_lse.size(), -INFINITY);
        bool stage_ok = true;
        size_t copied_bytes = 0;
        int windows = 0;
        const auto stage_start = std::chrono::steady_clock::now();
        for (size_t begin = 0; begin < selected_pages.size(); begin += slots) {
            const size_t count = std::min<size_t>(slots, selected_pages.size() - begin);
            std::fill(map.begin(), map.end(), -1);
            for (size_t slot = 0; slot < count; ++slot) {
                const int32_t page = selected_pages[begin + slot];
                const size_t source = (size_t) table[(size_t) page];
                map[(size_t) page] = (int32_t) slot;
                const uint8_t* src_k = (const uint8_t*) (fmt == 3 ? (const void*) kq.data() : (const void*) kh.data());
                const uint8_t* src_v = (const uint8_t*) (fmt == 3 ? (const void*) v4.data() : (const void*) vh.data());
                std::memcpy((uint8_t*) bounce_k + slot * k_bytes, src_k + source * k_bytes, k_bytes);
                std::memcpy((uint8_t*) bounce_v + slot * v_bytes, src_v + source * v_bytes, v_bytes);
                copied_bytes += k_bytes + v_bytes;
                if (fmt == 3) {
                    std::memcpy((uint8_t*) bounce_scale + slot * scale_bytes,
                                (const uint8_t*) ks.data() + source * scale_bytes, scale_bytes);
                    copied_bytes += scale_bytes;
                }
            }
            ck(cudaMemcpyAsync(stage_k, bounce_k, count * k_bytes, cudaMemcpyHostToDevice, nullptr), "stage K run");
            ck(cudaMemcpyAsync(stage_v, bounce_v, count * v_bytes, cudaMemcpyHostToDevice, nullptr), "stage V run");
            if (fmt == 3)
                ck(cudaMemcpyAsync(stage_scale, bounce_scale, count * scale_bytes, cudaMemcpyHostToDevice, nullptr),
                   "stage K scale run");
            for (int64_t qi = 0; qi < nq; ++qi) {
                int width = 0;
                for (int j = 0; j < steps[qi * k::kStepCount + k::kStepWidth]; ++j) {
                    const int32_t cell = ids[qi * cap + j];
                    if (map[(size_t) (cell / PS)] >= 0) part_ids[qi * cap + width++] = cell;
                }
                part_steps[qi * k::kStepCount + k::kStepWidth] = width;
                covered[(size_t) qi] += width;
            }
            ck(cudaMemcpy(stage_table, map.data(), map.size() * 4, cudaMemcpyHostToDevice), "stage map");
            ck(cudaMemcpy(stage_ids, part_ids.data(), part_ids.size() * 4, cudaMemcpyHostToDevice), "stage IDs");
            ck(cudaMemcpy(stage_steps, part_steps.data(), part_steps.size() * 4, cudaMemcpyHostToDevice), "stage steps");
            if (!k::qsa_prompt_attn_batch_lse(d_q, staged, stage_ids, stage_steps, cap, s,
                                              stage_out, stage_lse, nq, nullptr)) std::exit(2);
            ck(cudaDeviceSynchronize(), "bounded stage attention");
            ck(cudaMemcpy(part_out.data(), stage_out, part_out.size() * 4, cudaMemcpyDeviceToHost), "stage output down");
            ck(cudaMemcpy(part_lse.data(), stage_lse, part_lse.size() * 4, cudaMemcpyDeviceToHost), "stage LSE down");
            for (int64_t head = 0; head < nq * NH; ++head) {
                const double a = merged_lse[(size_t) head], b = part_lse[(size_t) head];
                if (!std::isfinite(b)) { if (b != -INFINITY) stage_ok = false; continue; }
                const double m = std::max(a, b);
                const double wa = std::isfinite(a) ? std::exp2(a - m) : 0.0;
                const double wb = std::exp2(b - m);
                for (int64_t d = 0; d < HD; ++d) {
                    const size_t at = (size_t) head * HD + d;
                    merged[at] = (wa * merged[at] + wb * part_out[at]) / (wa + wb);
                }
                merged_lse[(size_t) head] = m + std::log2(wa + wb);
            }
            ++windows;
        }
        const double stage_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - stage_start).count();
        double worst = 0, magnitude = 0;
        for (int64_t qi = 0; qi < nq; ++qi)
            if (covered[(size_t) qi] != steps[qi * k::kStepCount + k::kStepWidth]) stage_ok = false;
        for (size_t i = 0; i < nw.size(); ++i) {
            if (!std::isfinite(merged[i])) stage_ok = false;
            worst = std::max(worst, std::fabs(merged[i] - nw[i]));
            magnitude = std::max(magnitude, (double) std::fabs(nw[i]));
        }
        const bool pass = stage_ok && worst <= 3e-4 * std::max(magnitude, 1.0);
        ok_merge = ok_merge && pass;
        std::printf("%s %s bounded stage: %d windows, %d slots, %zu selected pages, %zu KV bytes copied, "
                    "%.3f ms wall, max difference %.3g (output scale %.3g)\n",
                    pass ? "PASS" : "FAIL", fmt == 3 ? "K8V4" : "FP16", windows, slots,
                    selected_pages.size(), copied_bytes, stage_ms, worst, magnitude);
        cudaFree(stage_k); cudaFree(stage_v); cudaFree(stage_scale); cudaFree(stage_table);
        cudaFree(stage_ids); cudaFree(stage_steps); cudaFree(stage_out); cudaFree(stage_lse);
        cudaFreeHost(bounce_k); cudaFreeHost(bounce_v); cudaFreeHost(bounce_scale);
    }
#endif
    // 1. FP64 reference on a sample of queries
    double err_old = 0, err_new = 0, ref_scale = 0;
    for (int64_t i = 0; i < nq; i += std::max<int64_t>(1, nq / 16)) {
        const int64_t w = steps[i * k::kStepCount + k::kStepWidth];
        const int32_t* sel = ids.data() + i * cap;
        for (int64_t h = 0; h < NH; ++h) {
            const int64_t kvh = h / (NH / NKV);
            std::vector<double> sco((size_t) w);
            double mx = -1e300;
            for (int64_t c = 0; c < w; ++c) {
                const int64_t cell = sel[c], row = ((int64_t) table[cell / PS] * NKV + kvh) * PS + cell % PS;
                double a = 0;
                for (int64_t d = 0; d < HD; ++d) {
                    const double kv = fmt == 2 ? q4v(k4, row, d)
                                      : (fmt == 1 || fmt == 3) ? (double) kq[row * HD + d] * h2f(ks[row * 4 + d / 64]) : h2f(kh[row * HD + d]);
                    a += (double) q[(i * NH + h) * HD + d] * kv;
                }
                sco[c] = a / 16.0;
                mx = std::max(mx, sco[c]);
            }
            double l = 0;
            for (auto& x : sco) { x = std::exp(x - mx); l += x; }
            for (int64_t d = 0; d < HD; ++d) {
                double a = 0;
                for (int64_t c = 0; c < w; ++c) {
                    const int64_t cell = sel[c], row = ((int64_t) table[cell / PS] * NKV + kvh) * PS + cell % PS;
                    const double vv = fmt == 3 ? h2f(f2h((float) q4v(v4, row, d)))
                                      : fmt == 2 ? q4v(v4, row, d)
                                      : fmt == 1 ? (double) vq[row * HD + d] * h2f(vs[row * 4 + d / 64]) : h2f(vh[row * HD + d]);
                    a += sco[c] * vv;
                }
                const double r = l > 0 ? a / l : 0;
                const size_t at = (size_t) ((i * NH + h) * HD + d);
                ref_scale = std::max(ref_scale, std::fabs(r));
                err_old = std::max(err_old, std::fabs(o[at] - r));
                err_new = std::max(err_new, std::fabs(nw[at] - r));
            }
        }
    }
    // 2. new vs old everywhere
    double diff = 0, scale = 0;
    bool finite = true;
    for (size_t i = 0; i < o.size(); ++i) {
        finite = finite && std::isfinite(o[i]) && std::isfinite(nw[i]);
        diff = std::max(diff, (double) std::fabs(o[i] - nw[i]));
        scale = std::max(scale, (double) std::fabs(o[i]));
    }
    bool ok4 = true;
#if defined(__HIP_PLATFORM_AMD__)
    {   // 4. the gated epilogue against the attention + gate_attn (bitwise)
        std::vector<float> qfull((size_t) (nq * NH * 2 * HD));
        for (auto& x : qfull) x = nd(rng) * 2.0f;
        const float* d_qfull = up(qfull);
        uint16_t *d_ref16 = nullptr, *d_got16 = nullptr;
        const size_t n16 = (size_t) (nq * NH * HD);
        ck(cudaMalloc(&d_ref16, n16 * 2), "malloc");
        ck(cudaMalloc(&d_got16, n16 * 2), "malloc");
        ck(cudaMemset(d_got16, 0, n16 * 2), "memset");
        strata::prefill::gate_attn(d_new, d_qfull, d_ref16, nq, nullptr, NH * HD);
        const bool taken = k::qsa_prompt_attn_batch_gated(d_q, pl, d_ids, d_steps, cap, s, d_qfull, d_got16, NH * HD, nq, nullptr);
        ck(cudaDeviceSynchronize(), "run");
        if (taken) {
            std::vector<uint16_t> ref(n16), got(n16);
            ck(cudaMemcpy(ref.data(), d_ref16, n16 * 2, cudaMemcpyDeviceToHost), "down");
            ck(cudaMemcpy(got.data(), d_got16, n16 * 2, cudaMemcpyDeviceToHost), "down");
            size_t bad = 0;
            for (size_t i = 0; i < n16; ++i) bad += ref[i] != got[i];
            ok4 = bad == 0;
            std::printf("%s gated epilogue: %zu of %zu FP16 values differ from the attention + gate_attn\n", ok4 ? "PASS" : "FAIL", bad, n16);
        } else {
            std::printf("gated epilogue: not taken here (the STRATA_PA_FAST=1 kernel only), not checked\n");
        }
        cudaFree((void*) d_qfull); cudaFree(d_ref16); cudaFree(d_got16);
    }
#endif
    // 3. speed
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float ms_old = 0, ms_new = 0;
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) old_run();
    cudaEventRecord(e1);
    ck(cudaEventSynchronize(e1), "time");
    cudaEventElapsedTime(&ms_old, e0, e1);
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) new_run();
    cudaEventRecord(e1);
    ck(cudaEventSynchronize(e1), "time");
    cudaEventElapsedTime(&ms_new, e0, e1);
    // Hybrid prompt V rounds to FP16; the decode kernel keeps the product in
    // FP32. Bound the prompt result tightly against the rounded FP64 oracle,
    // while allowing that existing storage-boundary difference against decode.
    const bool ok1 = finite && (fmt == 3 ? err_new <= 1e-5 * ref_scale
                                      : err_new <= std::max(4.0 * err_old, 1e-6 * ref_scale));
    const bool ok2 = finite && diff <= (fmt == 3 ? 1e-3 : 1e-4) * scale;
    std::printf("%s %s ctx %lld, %lld queries: vs FP64 old %.3g new %.3g (output scale %.3g); new vs old %.3g (%.2g of "
                "scale); %.3f -> %.3f ms per chunk (%.2fx)\n",
                ok1 && ok2 ? "PASS" : "FAIL", fmt == 3 ? "k8v4" : fmt == 2 ? "q4_0" : fmt == 1 ? "int8" : "fp16", (long long) ctx, (long long) nq, err_old,
                err_new, ref_scale, diff, diff / scale, ms_old / reps, ms_new / reps, ms_old / ms_new);
    cudaFree((void*) d_ids); cudaFree((void*) d_steps); cudaFree((void*) d_q); cudaFree(d_old); cudaFree(d_new);
    cudaFree(scratch);
    ck(cudaEventDestroy(e0), "event destroy");
    ck(cudaEventDestroy(e1), "event destroy");
    cudaFree((void*) pl.k_q); cudaFree((void*) pl.v_q); cudaFree((void*) pl.k_scale); cudaFree((void*) pl.v_scale);
    cudaFree((void*) pl.k_pool); cudaFree((void*) pl.v_pool); cudaFree((void*) pl.page_table);
    cudaFree((void*) pl.k_q4); cudaFree((void*) pl.v_q4);
    return ok1 && ok2 && ok4 && ok_merge ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
#if defined(__HIP_PLATFORM_AMD__)
    // S6: on AMD the kernel under test is the RDNA4 matrix-core one (opt-in in the engine); other cards skip
    {
        int dev = 0;
        hipDeviceProp_t prop{};
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess) return 2;
        if (std::strncmp(prop.gcnArchName, "gfx12", 5) != 0 && std::strncmp(prop.gcnArchName, "gfx11", 5) != 0) {
            std::printf("SKIP: %s is not gfx11 / gfx12 (the matrix-core prompt attention)\n", prop.gcnArchName);
            return 77;
        }
#if defined(_WIN32)
        _putenv_s("STRATA_HIP_WMMA", "1");
#else
        setenv("STRATA_HIP_WMMA", "1", 1);
#endif
    }
#endif
    const int64_t ctx = argc > 1 ? std::atoll(argv[1]) : 32768;
    const int64_t nq = argc > 2 ? std::atoll(argv[2]) : 2048;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 5;
    if (ctx <= 0 || nq <= 0 || nq > ctx || reps <= 0) {
        std::fprintf(stderr, "require context >= queries > 0 and reps > 0\n");
        return 2;
    }
    if (argc > 4) {
        if (std::strcmp(argv[4], "0") && std::strcmp(argv[4], "1") && std::strcmp(argv[4], "3")) return 2;
        return run(std::atoi(argv[4]), ctx, nq, reps);
    }
    if (std::getenv("QA_OUTPUT")) {
        std::fprintf(stderr, "QA_OUTPUT requires a single format (fourth argument 1 or 3)\n");
        return 2;
    }
    int fails = 0;
    fails += run(1, ctx, nq, reps);
#if !defined(__HIP_PLATFORM_AMD__)
    fails += run(3, ctx, nq, reps);
    fails += run(3, 33, std::min<int64_t>(nq, 33), reps); // chunk and ragged tails
    fails += run(0, ctx, nq, reps);   // FP16 KV: the RDNA4 kernel takes int8 KV only
    fails += run(2, ctx, nq, reps);   // Q4_0 KV (mode 4)
    fails += run(2, 1500, std::min<int64_t>(nq, 1500), reps);
#endif
    fails += run(1, 1500, std::min<int64_t>(nq, 1500), reps);   // short context: the selection is every cell
    fails += run(1, 2100, std::min<int64_t>(nq, 256), reps);    // the identity-to-sparse edge
    std::printf("FAILURES: %d\n", fails);
    return fails;
}
