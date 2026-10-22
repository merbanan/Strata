// Linux CUDA diagnostic interposer, not a production memory policy.
// STRATA_GRAPH_MEMORY_LOG=path (or '-' for stderr) enables lifecycle records.
// Free-memory deltas are observations, not exact ownership/accounting: CUDA
// pools, queued work and other GPU users can affect them. No extra GPU sync.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {
struct Memory {
    int device = -1, status = -1;
    int64_t free = -1, total = -1;
};
struct Entry {
    uint64_t id = 0, attempts = 0, submitted = 0;
    int64_t nodes = -1, kernels = -1;
    int inspected = 0;
};
struct Registry {
    std::mutex lock, output_lock;
    std::unordered_map<cudaGraphExec_t, Entry> graphs;
    uint64_t next_id = 0;
    bool header = false;
};
void exit_snapshot();
Registry& registry() {
    static Registry* value = [] {
        auto* result = new Registry;
        std::atexit(exit_snapshot);
        return result;
    }();
    return *value;
}
const char* destination() {
    static const char* path = std::getenv("STRATA_GRAPH_MEMORY_LOG");
    return path && *path ? path : nullptr;
}
template<class T> T next(const char* name) {
    auto value = reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
    if (!value) { std::fprintf(stderr, "graph memory trace: missing %s\n", name); std::abort(); }
    return value;
}
Memory memory() {
    Memory result;
    result.status = (int) cudaGetDevice(&result.device);
    if (result.status != (int) cudaSuccess) return result;
    size_t free = 0, total = 0;
    result.status = (int) cudaMemGetInfo(&free, &total);
    if (result.status == (int) cudaSuccess) { result.free = (int64_t) free; result.total = (int64_t) total; }
    return result;
}
void inspect(cudaGraph_t graph, Entry& entry) {
    size_t count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &count) != cudaSuccess) return;
    std::vector<cudaGraphNode_t> nodes(count);
    if (count && cudaGraphGetNodes(graph, nodes.data(), &count) != cudaSuccess) return;
    entry.nodes = (int64_t) count;
    entry.kernels = 0;
    for (size_t i = 0; i < count; ++i) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(nodes[i], &type) != cudaSuccess) { entry.kernels = -1; return; }
        if (type == cudaGraphNodeTypeKernel) ++entry.kernels;
    }
    // Counts are top-level nodes; child graphs are not flattened or attributed.
    entry.inspected = 1;
}
Entry get(cudaGraphExec_t graph) {
    auto& r = registry();
    std::lock_guard<std::mutex> guard(r.lock);
    auto found = r.graphs.find(graph);
    return found == r.graphs.end() ? Entry{} : found->second;
}
void record(const char* event, cudaGraphExec_t graph, const Entry& entry, int status,
            const Memory& before = {}, const Memory& after = {}) {
    const char* path = destination();
    if (!path) return;
    auto& r = registry();
    size_t live;
    {
        std::lock_guard<std::mutex> guard(r.lock);
        live = r.graphs.size();
    }
    std::lock_guard<std::mutex> guard(r.output_lock);
    const bool console = std::strcmp(path, "-") == 0;
    FILE* output = console ? stderr : std::fopen(path, "a");
    if (!output) { std::fprintf(stderr, "graph memory trace: cannot open %s\n", path); std::abort(); }
    if (!r.header) {
        std::fprintf(output, "GRAPH_MEM_HEADER\ttime_ns\tevent\tid\texec\tstatus\tlive_execs\tlaunch_attempts"
                             "\tlaunches_submitted\ttop_nodes\ttop_kernel_nodes\tinspected\tbefore_device"
                             "\tbefore_status\tbefore_free\tbefore_total\tafter_device\tafter_status\tafter_free\tafter_total\n");
        r.header = true;
    }
    const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::fprintf(output, "GRAPH_MEM\t%lld\t%s\t%llu\t%p\t%d\t%zu\t%llu\t%llu\t%lld\t%lld\t%d"
                         "\t%d\t%d\t%lld\t%lld\t%d\t%d\t%lld\t%lld\n",
                 (long long) time, event, (unsigned long long) entry.id, (void*) graph, status, live,
                 (unsigned long long) entry.attempts, (unsigned long long) entry.submitted,
                 (long long) entry.nodes, (long long) entry.kernels, entry.inspected,
                 before.device, before.status, (long long) before.free, (long long) before.total,
                 after.device, after.status, (long long) after.free, (long long) after.total);
    std::fflush(output);
    if (!console) std::fclose(output);
}
void exit_snapshot() {
    auto& r = registry();
    std::vector<std::pair<cudaGraphExec_t, Entry>> remaining;
    {
        std::lock_guard<std::mutex> guard(r.lock);
        for (const auto& pair : r.graphs) remaining.push_back(pair);
    }
    // No CUDA calls during process teardown. Global destructors may run later;
    // these entries are retained at this snapshot, not evidence of a leak.
    for (const auto& pair : remaining) record("exit_snapshot", pair.first, pair.second, -1);
}
}

extern "C" cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t* output, cudaGraph_t graph,
                                                     unsigned long long flags) {
    using Fn = cudaError_t (*)(cudaGraphExec_t*, cudaGraph_t, unsigned long long);
    static auto real = next<Fn>("cudaGraphInstantiate");
    if (!destination()) return real(output, graph, flags);
    const auto before = memory();
    const auto status = real(output, graph, flags);
    Memory after;
    Entry entry;
    cudaGraphExec_t exec = nullptr;
    if (status == cudaSuccess) {
        after = memory();
        inspect(graph, entry);
        exec = *output;
        auto& r = registry();
        std::lock_guard<std::mutex> guard(r.lock);
        entry.id = ++r.next_id;
        r.graphs[exec] = entry;
    }
    record("instantiate", exec, entry, (int) status, before, after);
    return status;
}

extern "C" cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graph, cudaStream_t stream) {
    using Fn = cudaError_t (*)(cudaGraphExec_t, cudaStream_t);
    static auto real = next<Fn>("cudaGraphLaunch");
    if (!destination()) return real(graph, stream);
    bool first = false;
    {
        auto& r = registry();
        std::lock_guard<std::mutex> guard(r.lock);
        auto found = r.graphs.find(graph);
        if (found != r.graphs.end()) first = found->second.attempts++ == 0;
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    const bool sample = first && cudaStreamIsCapturing(stream, &capture) == cudaSuccess
                             && capture == cudaStreamCaptureStatusNone;
    const auto before = sample ? memory() : Memory{};
    const auto status = real(graph, stream);
    const auto after = sample && status == cudaSuccess ? memory() : Memory{};
    Entry entry;
    {
        auto& r = registry();
        std::lock_guard<std::mutex> guard(r.lock);
        auto found = r.graphs.find(graph);
        if (found != r.graphs.end()) {
            if (status == cudaSuccess) ++found->second.submitted;
            entry = found->second;
        }
    }
    if (first || status != cudaSuccess || entry.id == 0)
        record(first ? "first_launch" : "launch_untracked_or_failed", graph, entry, (int) status, before, after);
    return status;
}

extern "C" cudaError_t CUDARTAPI cudaGraphUpload(cudaGraphExec_t graph, cudaStream_t stream) {
    using Fn = cudaError_t (*)(cudaGraphExec_t, cudaStream_t);
    static auto real = next<Fn>("cudaGraphUpload");
    if (!destination()) return real(graph, stream);
    const auto before = memory();
    const auto status = real(graph, stream);
    const auto after = status == cudaSuccess ? memory() : Memory{};
    record("upload", graph, get(graph), (int) status, before, after);
    return status;
}

extern "C" cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t graph) {
    using Fn = cudaError_t (*)(cudaGraphExec_t);
    static auto real = next<Fn>("cudaGraphExecDestroy");
    if (!destination()) return real(graph);
    const auto entry = get(graph);
    const auto before = memory();
    const auto status = real(graph);
    const auto after = status == cudaSuccess ? memory() : Memory{};
    if (status == cudaSuccess) {
        auto& r = registry();
        std::lock_guard<std::mutex> guard(r.lock);
        r.graphs.erase(graph);
    }
    record("destroy", graph, entry, (int) status, before, after);
    return status;
}
