#define IMAGEPROCCUDADLL_EXPORTS
#include "ImageProcApi.h"
#include <atomic>
#include <mutex>

static std::atomic<bool> g_initialized{ false };
static IPC_Params g_params{};
static std::mutex g_mtx;

static int32_t ValidateParams(const IPC_Params* p)
{
    if (!p) return IPC_ERR_INVALIDARG;
    if (p->sizeBytes < sizeof(IPC_Params)) return IPC_ERR_INVALIDARG;
    if (p->version != 1) return IPC_ERR_INVALIDARG;
    return IPC_OK;
}

extern "C" {

    int32_t IPC_Init(int32_t gpuId, void* inSharedHandle, void* outSharedHandle)
    {
        // Phase 2-1 では引数を受け取るだけ（D3D11/OpenSharedResourceはまだやらない）
        (void)gpuId;
        (void)inSharedHandle;
        (void)outSharedHandle;

        std::lock_guard<std::mutex> lock(g_mtx);

        // デフォルトparams
        g_params.sizeBytes = sizeof(IPC_Params);
        g_params.version = 1;
        g_params.window = 4000;
        g_params.level = 2000;
        g_params.enableEdge = 0;

        g_initialized = true;
        return IPC_OK;
    }

    int32_t IPC_SetParams(const IPC_Params* p)
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        if (!g_initialized) return IPC_ERR_NOT_INITIALIZED;

        int32_t v = ValidateParams(p);
        if (v != IPC_OK) return v;

        g_params = *p;
        return IPC_OK;
    }

    int32_t IPC_Execute()
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        if (!g_initialized) return IPC_ERR_NOT_INITIALIZED;

        // Phase 2-1：何もしない（成功だけ返す）
        return IPC_OK;
    }

    int32_t IPC_Shutdown()
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        g_initialized = false;
        return IPC_OK;
    }

} // extern "C"