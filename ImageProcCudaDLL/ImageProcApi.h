#pragma once
#include <stdint.h>

#ifdef IMAGEPROCCUDADLL_EXPORTS
#define IPC_API __declspec(dllexport)
#else
#define IPC_API __declspec(dllimport)
#endif

// Forward declare (avoid including CUDA headers here)
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

        // WL/WW (16-bit domain)
        int32_t window;       // 1..65535
        int32_t level;        // 0..65535

        // flags
        int32_t enableEdge;   // 0: WL/WW, 1: edge(sobel)

        // reserved:
        // reserved[0] is used as enablePostFilter (0/1) in this implementation
        int32_t reserved[8];
    };
#pragma pack(pop)

    // Init: gpuId + shared handles (both must be D3D11 shared Texture2D with DXGI_FORMAT_R16_UINT)
    IPC_API int32_t __cdecl IPC_Init(int32_t gpuId, void* inSharedHandle, void* outSharedHandle);

    // Params
    IPC_API int32_t __cdecl IPC_SetParams(const IPC_Params* p);

    // Execute: Input(R16_UINT) -> Output(R16_UINT)
    IPC_API int32_t __cdecl IPC_Execute();

    // Shutdown
    IPC_API int32_t __cdecl IPC_Shutdown();


    // ---- CudaInterop.cu exported functions ----
    IPC_API int __cdecl CudaSetDeviceSafe(int gpuId);
    IPC_API int __cdecl CudaRegisterD3D11Texture(void* tex2D, cudaGraphicsResource** outRes);
    IPC_API int __cdecl CudaUnregister(cudaGraphicsResource* res);

    // Map both resources and get mapped cudaArrays (resources remain mapped!)
    IPC_API int __cdecl CudaMapGetArraysMapped(
        cudaGraphicsResource* inRes,
        cudaGraphicsResource* outRes,
        void** inArray,
        void** outArray);

    // Unmap both resources (must be called after processing)
    IPC_API int __cdecl CudaUnmapResources(
        cudaGraphicsResource* inRes,
        cudaGraphicsResource* outRes);

    // Process: in(R16_UINT array) -> out(R16_UINT array)
    // enablePostFilter: apply extra filter after first stage (demo: 3x3 box blur)
    IPC_API int __cdecl CudaProcessArrays_R16_To_R16(
        void* inArray,
        void* outArray,
        int w,
        int h,
        int window,
        int level,
        int enableEdge,
        int enablePostFilter);
} // extern "C"
