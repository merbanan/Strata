// Linux/CUDA diagnostic only. Counts grouped expert kernel launch shapes.
// Capture records are not executions; GRAPH/REPLAY records account for replay.
// Does not read device group counts, measure GPU time, or synchronize streams.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
using Shapes = std::map<std::string, size_t>;
struct Registry {
    std::mutex lock;
    std::unordered_map<const void*, std::string> names;
    std::unordered_map<cudaGraphExec_t, Shapes> graphs;
};
Registry& registry() { static auto* value = new Registry; return *value; }
std::mutex& log_lock() { static auto* value = new std::mutex; return *value; }
template<class T> T next(const char* name) {
    auto value = reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
    if (!value) { std::fprintf(stderr, "expert trace: missing %s\n", name); std::abort(); }
    return value;
}
bool enabled() { return std::getenv("STRATA_EXPERT_LAUNCH_LOG") != nullptr; }
void log(const std::string& text) {
    const char* path = std::getenv("STRATA_EXPERT_LAUNCH_LOG");
    if (!path) return;
    std::lock_guard<std::mutex> guard(log_lock());
    FILE* file = std::fopen(path, "a");
    if (!file) { std::perror("expert trace log"); std::abort(); }
    const bool ok = std::fwrite(text.data(), 1, text.size(), file) == text.size();
    const int closed = std::fclose(file);
    if (!ok || closed) { std::fprintf(stderr, "expert trace: log write failed\n"); std::abort(); }
}
std::string name_of(const void* fn) {
    auto& r = registry(); std::lock_guard<std::mutex> guard(r.lock);
    const auto it = r.names.find(fn);
    return it == r.names.end() ? std::string() : it->second;
}
std::string shape(const std::string& name, dim3 grid, dim3 block, size_t shared) {
    std::ostringstream out;
    out << name << '\t' << grid.x << '\t' << grid.y << '\t' << grid.z
        << '\t' << block.x << '\t' << block.y << '\t' << block.z << '\t' << shared;
    return out.str();
}
bool graph_shapes(cudaGraph_t graph, Shapes& shapes) {
    size_t count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &count) != cudaSuccess) return false;
    std::vector<cudaGraphNode_t> nodes(count);
    if (count && cudaGraphGetNodes(graph, nodes.data(), &count) != cudaSuccess) return false;
    for (size_t i = 0; i < count; ++i) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(nodes[i], &type) != cudaSuccess) return false;
        if (type == cudaGraphNodeTypeGraph) {
            cudaGraph_t child;
            if (cudaGraphChildGraphNodeGetGraph(nodes[i], &child) != cudaSuccess || !graph_shapes(child, shapes)) return false;
        } else if (type == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params{};
            if (cudaGraphKernelNodeGetParams(nodes[i], &params) != cudaSuccess) return false;
            auto name = name_of(params.func);
            if (!name.empty()) ++shapes[shape(name, params.gridDim, params.blockDim, params.sharedMemBytes)];
        } else if (type == cudaGraphNodeTypeConditional) {
            return false; // Static multiplicity cannot describe conditional execution.
        }
    }
    return true;
}
void launch_log(const std::string& name, dim3 grid, dim3 block, size_t bytes,
                cudaError_t status, cudaError_t queried, cudaStreamCaptureStatus capture) {
    std::ostringstream out;
    out << "LAUNCH\t" << int(status) << '\t' << int(queried) << '\t' << int(capture)
        << '\t' << shape(name, grid, block, bytes) << '\n';
    log(out.str());
}
}

extern "C" void CUDARTAPI __cudaRegisterFunction(void** handle, const char* host,
    char* device, const char* name, int limit, uint3* tid, uint3* bid,
    dim3* bdim, dim3* gdim, int* warp) {
    using Fn = void (*)(void**, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*);
    static auto real = next<Fn>("__cudaRegisterFunction");
    real(handle, host, device, name, limit, tid, bid, bdim, gdim, warp);
    if (name && (std::strstr(name, "native_gu_") || std::strstr(name, "native_down_") ||
                 std::strstr(name, "swiglu_q8_1_entries_kernel"))) {
        auto& r = registry(); std::lock_guard<std::mutex> guard(r.lock);
        r.names[host] = name;
    }
}

extern "C" cudaError_t CUDARTAPI cudaLaunchKernel(const void* fn, dim3 grid, dim3 block,
    void** args, size_t bytes, cudaStream_t stream) {
    using Fn = cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
    static auto real = next<Fn>("cudaLaunchKernel");
    const auto name = enabled() ? name_of(fn) : std::string();
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    const auto queried = name.empty() ? cudaSuccess : cudaStreamIsCapturing(stream, &capture);
    const auto status = real(fn, grid, block, args, bytes, stream);
    if (!name.empty()) launch_log(name, grid, block, bytes, status, queried, capture);
    return status;
}

extern "C" cudaError_t CUDARTAPI cudaLaunchKernelExC(const cudaLaunchConfig_t* config,
    const void* fn, void** args) {
    using Fn = cudaError_t (*)(const cudaLaunchConfig_t*, const void*, void**);
    static auto real = next<Fn>("cudaLaunchKernelExC");
    const auto name = enabled() && config ? name_of(fn) : std::string();
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    const auto queried = name.empty() ? cudaSuccess : cudaStreamIsCapturing(config->stream, &capture);
    const auto status = real(config, fn, args);
    if (!name.empty()) launch_log(name, config->gridDim, config->blockDim, config->dynamicSmemBytes, status, queried, capture);
    return status;
}

extern "C" cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t* output,
    cudaGraph_t graph, unsigned long long flags) {
    using Fn = cudaError_t (*)(cudaGraphExec_t*, cudaGraph_t, unsigned long long);
    static auto real = next<Fn>("cudaGraphInstantiate");
    const auto status = real(output, graph, flags);
    if (enabled()) {
        Shapes shapes;
        const bool inspected = status == cudaSuccess && graph_shapes(graph, shapes);
        cudaGraphExec_t exec = status == cudaSuccess ? *output : nullptr;
        if (inspected) {
            auto& r = registry(); std::lock_guard<std::mutex> guard(r.lock);
            r.graphs[exec] = shapes;
        }
        std::ostringstream out;
        out << "GRAPH\t" << (void*)exec << '\t' << int(status) << '\t' << inspected << '\t' << shapes.size() << '\n';
        for (const auto& item : shapes) out << "NODE\t" << (void*)exec << '\t' << item.second << '\t' << item.first << '\n';
        log(out.str());
    }
    return status;
}

extern "C" cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graph, cudaStream_t stream) {
    using Fn = cudaError_t (*)(cudaGraphExec_t, cudaStream_t);
    static auto real = next<Fn>("cudaGraphLaunch");
    const auto status = real(graph, stream);
    if (enabled()) {
        bool known;
        { auto& r = registry(); std::lock_guard<std::mutex> guard(r.lock); known = r.graphs.count(graph) != 0; }
        std::ostringstream out;
        out << "REPLAY\t" << (void*)graph << '\t' << int(status) << '\t' << known << '\n';
        log(out.str());
    }
    return status;
}

extern "C" cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t graph) {
    using Fn = cudaError_t (*)(cudaGraphExec_t);
    static auto real = next<Fn>("cudaGraphExecDestroy");
    const auto status = real(graph);
    if (status == cudaSuccess) {
        auto& r = registry(); std::lock_guard<std::mutex> guard(r.lock);
        r.graphs.erase(graph);
    }
    if (enabled()) {
        std::ostringstream out; out << "DESTROY\t" << (void*)graph << '\t' << int(status) << '\n'; log(out.str());
    }
    return status;
}
