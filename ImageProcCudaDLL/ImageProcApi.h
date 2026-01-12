#pragma once
#include <stdint.h>

#ifdef IMAGEPROCCUDADLL_EXPORTS
#define IPC_API __declspec(dllexport)
#else
#define IPC_API __declspec(dllimport)
#endif

struct cudaGraphicsResource;

extern "C" {

    enum IPC_Result : int32_t
    {
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

        int32_t window;       // 1..65535
        int32_t level;        // 0..65535
        int32_t enableEdge;   // 0: WL/WW, 1: sobel

        // reserved[0] = enablePostFilter (0/1)
        int32_t reserved[8];
    };
#pragma pack(pop)

    // Single shared texture handle (DXGI_FORMAT_R16_UINT)
    IPC_API int32_t __cdecl IPC_Init(int32_t gpuId, void* ioSharedHandle);
    IPC_API int32_t __cdecl IPC_SetParams(const IPC_Params* p);
    IPC_API int32_t __cdecl IPC_Execute();
    IPC_API int32_t __cdecl IPC_Shutdown();

    // CUDA interop
    IPC_API int __cdecl CudaSetDeviceSafe(int gpuId);
    IPC_API int __cdecl CudaRegisterD3D11Texture(void* tex2D, cudaGraphicsResource** outRes);
    IPC_API int __cdecl CudaUnregister(cudaGraphicsResource* res);

    // Map single resource and get mapped cudaArray (resource remains mapped)
    IPC_API int __cdecl CudaMapGetArrayMapped(cudaGraphicsResource* ioRes, void** ioArray);
    IPC_API int __cdecl CudaUnmapResource(cudaGraphicsResource* ioRes);

    // In-place processing using intermediate array(s) inside CUDA
    IPC_API int __cdecl CudaProcessArray_R16_Inplace(
        void* ioArray,
        int w,
        int h,
        int window,
        int level,
        int enableEdge,
        int enablePostFilter);

    IPC_API int __cdecl CudaReleaseCache();
}
