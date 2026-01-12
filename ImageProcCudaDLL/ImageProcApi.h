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

        // Window/Level (16-bit domain)
        int32_t window;       // 1..65535
        int32_t level;        // 0..65535

        // 0: WL/WW only, 1: edge output (simple Sobel)
        int32_t enableEdge;

        int32_t reserved0;
        int32_t reserved1;
        int32_t reserved2;
        int32_t reserved3;
    };
#pragma pack(pop)

    // ---- Public DLL API (called from C#) ----

    // Initialize interop.
    // inSharedHandle/outSharedHandle must be handles to D3D11 shared textures:
    //   input  = DXGI_FORMAT_R16_UINT
    //   output = DXGI_FORMAT_R16_UINT
    IPC_API int32_t __cdecl IPC_Init(int32_t gpuId, void* inSharedHandle, void* outSharedHandle);

    IPC_API int32_t __cdecl IPC_SetParams(const IPC_Params* p);

    // Execute processing: Input(R16_UINT) -> Output(R16_UINT)
    IPC_API int32_t __cdecl IPC_Execute();

    IPC_API int32_t __cdecl IPC_Shutdown();

    // ---- CUDA helpers (implemented in CudaInterop.cu) ----
    IPC_API int CudaSetDeviceSafe(int gpuId);
    IPC_API int CudaRegisterD3D11Texture(void* tex2D, cudaGraphicsResource** outRes);
    IPC_API int CudaUnregister(cudaGraphicsResource* res);

    // Map and get arrays (keeps mapped until CudaUnmapResources)
    IPC_API int CudaMapGetArraysMapped(
        cudaGraphicsResource* inRes,
        cudaGraphicsResource* outRes,
        void** inArray,
        void** outArray);

    IPC_API int CudaUnmapResources(
        cudaGraphicsResource* inRes,
        cudaGraphicsResource* outRes);

    // Process: in(R16_UINT array) -> out(R16_UINT array)
    IPC_API int CudaProcessArrays_R16_To_R16(
        void* inArray,
        void* outArray,
        int w,
        int h,
        int window,
        int level,
        int enableEdge);

} // extern "C"
