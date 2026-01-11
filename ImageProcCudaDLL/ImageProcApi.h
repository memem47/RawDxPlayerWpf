#pragma once
#include <stdint.h>

#ifdef IMAGEPROCCUDADLL_EXPORTS
#define IPC_API __declspec(dllexport)
#else
#define IPC_API __declspec(dllimport)
#endif

extern "C" {

    enum IPC_Result : int32_t {
        IPC_OK = 0,
        IPC_ERR_INVALIDARG = 1,
        IPC_ERR_NOT_INITIALIZED = 2,
        IPC_ERR_INTERNAL = 3
    };

#pragma pack(push, 1)
    struct IPC_Params
    {
        uint32_t sizeBytes;   // sizeof(IPC_Params)
        uint32_t version;     // 1

        // WL/WW
        int32_t window;
        int32_t level;

        // effect flags
        int32_t enableEdge;

        int32_t reserved[8];
    };
#pragma pack(pop)

    // A: init (GPU id + shared handles)
    IPC_API int32_t IPC_Init(int32_t gpuId, void* inSharedHandle, void* outSharedHandle);

    // B: params
    IPC_API int32_t IPC_SetParams(const IPC_Params* p);

    // C: execute
    IPC_API int32_t IPC_Execute();

    // D: shutdown
    IPC_API int32_t IPC_Shutdown();
}

// ---- CudaInterop.cu exported functions ----
struct cudaGraphicsResource;

extern "C" int CudaSetDeviceSafe(int gpuId);
extern "C" int CudaRegisterD3D11Texture(void* tex2D, cudaGraphicsResource** outRes);
extern "C" int CudaUnregister(cudaGraphicsResource* res);

// Map both resources and get mapped cudaArrays (resources remain mapped!)
extern "C" int CudaMapGetArraysMapped(
    cudaGraphicsResource* inRes,
    cudaGraphicsResource* outRes,
    void** inArray,
    void** outArray);

// Unmap both resources (must be called after processing)
extern "C" int CudaUnmapResources(
    cudaGraphicsResource* inRes,
    cudaGraphicsResource* outRes);

// Process: in(R16_UINT array) -> out(BGRA array)
extern "C" int CudaProcessArrays_R16_To_BGRA(
    void* inArray,
    void* outArray,
    int w,
    int h,
    int window,
    int level,
    int enableEdge);
