// Local CUDA diagnostic for the real MtpDrafter loan/restore methods.
// Compile this test only with -fno-access-control and --wrap=cudaMemcpy,
// --wrap=cudaMallocHost, --wrap=cudaStreamSynchronize. No production test API.
#include "strata/core/mtp.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using strata::core::MtpDrafter;
static MtpDrafter* active;
static int d2h_count, h2d_count;
static bool fail_alloc, fail_d2h, fail_h2d, fail_idle;
extern "C" cudaError_t __real_cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
extern "C" cudaError_t __real_cudaMallocHost(void**, size_t);
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t);

extern "C" cudaError_t __wrap_cudaMallocHost(void** p, size_t n) {
    if (fail_alloc) { fail_alloc = false; *p = nullptr; return cudaErrorMemoryAllocation; }
    return __real_cudaMallocHost(p, n);
}
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
    if (fail_idle) { fail_idle = false; return cudaErrorUnknown; }
    return __real_cudaStreamSynchronize(stream);
}
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind kind) {
    if (active && dst == active->experts_host_ && src == active->experts_ && kind == cudaMemcpyDeviceToHost) {
        ++d2h_count;
        if (fail_d2h) {
            fail_d2h = false;
            // A failed transfer may have partially changed the destination.
            std::memset(dst, 0xa5, n / 2);
            return cudaErrorUnknown;
        }
    }
    if (active && dst == active->experts_ && src == active->experts_host_ && kind == cudaMemcpyHostToDevice) {
        ++h2d_count;
        if (fail_h2d) { fail_h2d = false; return cudaErrorUnknown; }
    }
    return __real_cudaMemcpy(dst, src, n, kind);
}
static void require(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", what); std::exit(1); }
}
static void ck(cudaError_t e) { require(e == cudaSuccess, cudaGetErrorString(e)); }

int main() {
    const char* flag = std::getenv("STRATA_MTP_EXPERT_SNAPSHOT");
    const bool reuse = flag && std::strcmp(flag, "1") == 0;
    strata::core::ModelGeometry geometry{};
    geometry.n_expert = 8;
    const size_t bytes = geometry.n_expert * strata::kernels::cpu::BLOB;
    std::vector<unsigned char> original(bytes), result(bytes);
    for (size_t i = 0; i < bytes; ++i) original[i] = (unsigned char) ((i * 131 + i / 257) & 255);
    MtpDrafter drafter;
    active = &drafter;
    drafter.g_ = &geometry;
    ck(cudaGetDevice(&drafter.device_));
    ck(cudaMalloc((void**) &drafter.experts_, bytes));
    ck(cudaStreamCreate(&drafter.cs_));
    ck(cudaMemcpy(drafter.experts_, original.data(), bytes, cudaMemcpyHostToDevice));
    void* const address = drafter.experts_base();
    std::string error;
    fail_alloc = true;
    require(!drafter.experts_lend(error) && drafter.experts_resident() && !drafter.experts_host_valid_, "allocation failure");
    fail_d2h = true;
    require(!drafter.experts_lend(error) && drafter.experts_resident() && !drafter.experts_host_valid_, "failed copy cannot validate snapshot");
    require(drafter.experts_lend(error) && drafter.experts_host_valid_, "copy retry");
    require(std::memcmp(drafter.experts_host_, original.data(), bytes) == 0, "snapshot bytes");
    require(!drafter.decode_ready(error), "decode blocked while lent");
    const int first_count = d2h_count;
    require(drafter.experts_lend(error) && d2h_count == first_count, "already lent is idempotent");
    ck(cudaMemset(drafter.experts_, 0x3c, bytes));
    fail_h2d = true;
    require(!drafter.experts_reclaim(error) && !drafter.experts_resident(), "failed reclaim stays lent");
    require(drafter.experts_reclaim(error) && drafter.decode_ready(error), "reclaim retry");
    const int restored_count = h2d_count;
    require(drafter.experts_reclaim(error) && h2d_count == restored_count, "already reclaimed is idempotent");

    // Capture the expert address before repeated loans; replay must read the
    // restored bytes at that same address after every destructive room use.
    void* readback = nullptr;
    ck(cudaMalloc(&readback, bytes));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(drafter.cs_, cudaStreamCaptureModeGlobal));
    ck(cudaMemcpyAsync(readback, drafter.experts_, bytes, cudaMemcpyDeviceToDevice, drafter.cs_));
    ck(cudaStreamEndCapture(drafter.cs_, &graph));
    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int cycle = 0; cycle < 5; ++cycle) {
        require(drafter.experts_lend(error), "repeat loan");
        ck(cudaMemset(drafter.experts_, 0x80 + cycle, bytes));
        require(drafter.experts_reclaim(error), "repeat reclaim");
        require(drafter.experts_base() == address, "graph address unchanged");
        ck(cudaGraphLaunch(exec, drafter.cs_));
        ck(cudaStreamSynchronize(drafter.cs_));
        ck(cudaMemcpy(result.data(), readback, bytes, cudaMemcpyDeviceToHost));
        require(result == original, "graph reads original bytes after reuse");
    }
    require(d2h_count == first_count + (reuse ? 0 : 5), "expected repeated-copy count");
    drafter.chain_live_ = true;
    require(!drafter.experts_lend(error) && drafter.experts_resident(), "live chain cannot lend cached snapshot");
    drafter.chain_live_ = false;
    fail_idle = true;
    require(!drafter.experts_lend(error) && drafter.experts_resident(), "failed idle cannot lend cached snapshot");
    {
        MtpDrafter shared;
        shared.owns_weights_ = false;
        shared.g_ = &geometry;
        shared.experts_ = drafter.experts_;
        require(!shared.experts_lend(error) && shared.experts_bytes() == 0, "shared drafter cannot lend owner weights");
    }
    ++drafter.experts_host_bytes_;
    require(!drafter.experts_lend(error) && drafter.experts_resident(), "reject mismatched snapshot size");
    --drafter.experts_host_bytes_;

    // Exercise invalidation through the public loader, failing before any
    // replacement weights are allocated. A stale snapshot must not survive.
    strata::core::SessionState session;
    require(!drafter.load("/nonexistent/strata-snapshot-test", geometry, session, 1, error), "expected missing load");
    require(!drafter.experts_host_valid_, "load attempt invalidates snapshot");
    for (auto& x : original) x ^= 0x5a;
    ck(cudaMemcpy(drafter.experts_, original.data(), bytes, cudaMemcpyHostToDevice));
    const int before_reload = d2h_count;
    require(drafter.experts_lend(error) && d2h_count == before_reload + 1, "invalidated snapshot recopied");
    ck(cudaMemset(drafter.experts_, 0, bytes));
    require(drafter.experts_reclaim(error), "reclaim new identity");
    ck(cudaMemcpy(result.data(), drafter.experts_, bytes, cudaMemcpyDeviceToHost));
    require(result == original, "new snapshot bytes");
    ck(cudaGraphExecDestroy(exec));
    ck(cudaGraphDestroy(graph));
    ck(cudaFree(readback));
    std::printf("PASS snapshot reuse=%d bytes=%zu D2H=%d H2D=%d; failures, ownership, invalidation, five graph cycles\n",
                reuse, bytes, d2h_count, h2d_count);
    active = nullptr;
}
