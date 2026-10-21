// Linux-only diagnostic interposer. Records successful direct attention launches;
// it does not measure GPU time or performance counters. Arguments pass unchanged.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {
struct Registry {
    std::mutex lock;
    std::unordered_map<const void*, std::string> names;
};
Registry& registry() { static Registry* r = new Registry; return *r; }
template<class T> T next(const char* name) {
    auto p = reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
    if (!p) { std::fprintf(stderr, "attention trace: missing %s\n", name); std::abort(); }
    return p;
}
}

extern "C" void CUDARTAPI __cudaRegisterFunction(void** handle, const char* host,
    char* device, const char* name, int limit, uint3* tid, uint3* bid,
    dim3* bdim, dim3* gdim, int* warp) {
    using Fn = void (*)(void**, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*);
    static auto real = next<Fn>("__cudaRegisterFunction");
    real(handle, host, device, name, limit, tid, bid, bdim, gdim, warp);
    if (name && std::strstr(name, "prompt_attn")) {
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
        if (const char* path = std::getenv("STRATA_ATTENTION_LAUNCH_LOG")) {
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
