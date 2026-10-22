// Linux-only diagnostic interposer. Records router/Q8 fusion launches and graph captures;
// it does not measure GPU time or performance counters. Arguments pass unchanged.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
struct Registry {
    std::mutex lock;
    std::unordered_map<const void*, std::string> names;
    using Shapes = std::map<std::pair<int, int>, size_t>;
    std::unordered_map<cudaGraphExec_t, Shapes> graphs;
};
Registry& registry() { static Registry* r = new Registry; return *r; }
template<class T> T next(const char* name) {
    auto p = reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
    if (!p) { std::fprintf(stderr, "router/Q8 trace: missing %s\n", name); std::abort(); }
    return p;
}
bool graph_shapes(cudaGraph_t graph, Registry::Shapes& shapes) {
    size_t count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &count) != cudaSuccess) return false;
    std::vector<cudaGraphNode_t> nodes(count);
    if (count && cudaGraphGetNodes(graph, nodes.data(), &count) != cudaSuccess) return false;
    for (size_t i = 0; i < count; ++i) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(nodes[i], &type) != cudaSuccess) return false;
        if (type == cudaGraphNodeTypeGraph) {
            cudaGraph_t child;
            if (cudaGraphChildGraphNodeGetGraph(nodes[i], &child) != cudaSuccess || !graph_shapes(child, shapes))
                return false;
        } else if (type == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params{};
            if (cudaGraphKernelNodeGetParams(nodes[i], &params) != cudaSuccess) return false;
            bool match;
            {
                auto& r = registry(); std::lock_guard<std::mutex> guard(r.lock);
                match = r.names.count(params.func) != 0;
            }
            if (match) {
                // route_q8_fused's fixed signature: five pointer arguments,
                // then int rows and int columns. This tracer is specific to it.
                if (!params.kernelParams || !params.kernelParams[5] || !params.kernelParams[6]) return false;
                int rows, cols;
                std::memcpy(&rows, params.kernelParams[5], sizeof(rows));
                std::memcpy(&cols, params.kernelParams[6], sizeof(cols));
                ++shapes[{rows, cols}];
            }
        }
    }
    return true;
}
void replay_log(const char* event, cudaGraphExec_t graph, int status, bool inspected, const Registry::Shapes& shapes) {
    if (const char* path = std::getenv("STRATA_ROUTE_Q8_REPLAY_LOG")) {
        if (FILE* file = std::fopen(path, "a")) {
            std::fprintf(file, "%s\t%p\t%d\t%d", event, (void*) graph, status, inspected ? 1 : 0);
            for (const auto& entry : shapes)
                std::fprintf(file, "\t%d:%d:%zu", entry.first.first, entry.first.second, entry.second);
            std::fprintf(file, "\n");
            std::fclose(file);
        }
    }
}
}

extern "C" void CUDARTAPI __cudaRegisterFunction(void** handle, const char* host,
    char* device, const char* name, int limit, uint3* tid, uint3* bid,
    dim3* bdim, dim3* gdim, int* warp) {
    using Fn = void (*)(void**, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*);
    static auto real = next<Fn>("__cudaRegisterFunction");
    real(handle, host, device, name, limit, tid, bid, bdim, gdim, warp);
    if (name && std::strstr(name, "route_q8_fused")) {
        auto& r = registry(); std::lock_guard<std::mutex> guard(r.lock);
        r.names[host] = name;
    }
}

extern "C" cudaError_t CUDARTAPI cudaLaunchKernel(const void* fn, dim3 grid, dim3 block,
    void** args, size_t bytes, cudaStream_t stream) {
    using Fn = cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
    static auto real = next<Fn>("cudaLaunchKernel");
    std::string name;
    {
        auto& r = registry(); std::lock_guard<std::mutex> guard(r.lock);
        auto it = r.names.find(fn); if (it != r.names.end()) name = it->second;
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    const auto queried = name.empty() ? cudaSuccess : cudaStreamIsCapturing(stream, &capture);
    const auto status = real(fn, grid, block, args, bytes, stream);
    if (!name.empty()) {
        if (const char* path = std::getenv("STRATA_ROUTE_Q8_LAUNCH_LOG")) {
            if (FILE* f = std::fopen(path, "a")) {
                std::fprintf(f, "%s\t%d\t%d\t%d\t%u\t%u\t%u\t%u\t%u\t%u\t%zu\n",
                    name.c_str(), (int) status, (int) queried, (int) capture,
                    grid.x, grid.y, grid.z, block.x, block.y, block.z, bytes);
                std::fclose(f);
            }
        }
    }
    return status;
}

extern "C" cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t* output, cudaGraph_t graph,
                                                     unsigned long long flags) {
    using Fn = cudaError_t (*)(cudaGraphExec_t*, cudaGraph_t, unsigned long long);
    static auto real = next<Fn>("cudaGraphInstantiate");
    const auto status = real(output, graph, flags);
    if (status == cudaSuccess && std::getenv("STRATA_ROUTE_Q8_REPLAY_LOG")) {
        Registry::Shapes shapes;
        const bool inspected = graph_shapes(graph, shapes);
        if (inspected) {
            auto& r = registry(); std::lock_guard<std::mutex> guard(r.lock);
            r.graphs[*output] = shapes;
        }
        replay_log("GRAPH", *output, (int) status, inspected, shapes);
    }
    return status;
}

extern "C" cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graph, cudaStream_t stream) {
    using Fn = cudaError_t (*)(cudaGraphExec_t, cudaStream_t);
    static auto real = next<Fn>("cudaGraphLaunch");
    const auto status = real(graph, stream);
    if (std::getenv("STRATA_ROUTE_Q8_REPLAY_LOG")) {
        Registry::Shapes shapes;
        bool inspected;
        {
            auto& r = registry(); std::lock_guard<std::mutex> guard(r.lock);
            auto it = r.graphs.find(graph);
            inspected = it != r.graphs.end();
            if (inspected) shapes = it->second;
        }
        replay_log("REPLAY", graph, (int) status, inspected, shapes);
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
    return status;
}
