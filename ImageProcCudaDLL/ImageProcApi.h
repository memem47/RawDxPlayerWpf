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
        IPC_ERR_INTERNAL = 3,
        IPC_ERR_INVALID_ARG = 4,
        IPC_ERR_INVALID_STATE = 5
    };


#pragma pack(push, 1)
    struct IPC_Params
    {
        int32_t width;
        int32_t height;
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
    IPC_API int32_t __cdecl IPC_InitWithIoBuffer(int32_t gpuId, void* ioBufferPtr);

    // Params
    IPC_API int32_t __cdecl IPC_SetParams(const IPC_Params* p);
    IPC_API int32_t __cdecl IPC_Execute();
    IPC_API int32_t __cdecl IPC_Shutdown();

    // CUDA interop
    IPC_API int32_t __cdecl IPC_ReadbackRaw16(void* dst, int32_t dstBytes);
    IPC_API int32_t __cdecl IPC_ReadbackRaw16FromBuffer(void* dst, int32_t dstBytes);

    // ---- CudaInterop.cu exported functions ----
    IPC_API int __cdecl CudaSetDeviceSafe(int gpuId);
    IPC_API int __cdecl CudaRegisterD3D11Texture(void* tex2D, cudaGraphicsResource** outRes);
    IPC_API int __cdecl CudaUnregister(cudaGraphicsResource* res);

    IPC_API int __cdecl CudaRegisterD3D11Buffer(void* buf, cudaGraphicsResource** outRes);

    // Map single resource and get mapped cudaArray (resource remains mapped)
    IPC_API int __cdecl CudaMapGetArrayMapped(cudaGraphicsResource* ioRes, void** ioArray);

    // Map single resource and get mapped device pointer (resource remains mapped)
    IPC_API int __cdecl CudaMapGetPointerMapped(cudaGraphicsResource* ioRes, void** devPtr, size_t* outBytes);

    
    IPC_API int __cdecl CudaUnmapResource(cudaGraphicsResource* ioRes);
    
    // In-place processing using intermediate array(s) inside CUDA
    IPC_API int __cdecl CudaProcessArray_R16_Inplace(
        void* ioArray,
        int w,
        int h,
        int window,
        int level,
        int enableEdge,
        int enableBlur,
        int enableInvert,
        int enableThreshold,
        int thresholdValue);
    
    // In-place processing for linear buffer (uint16_t*)
    IPC_API int __cdecl CudaProcessBuffer_R16_Inplace(
        void* ioDevPtr,
        int w,
        int h,
        int window,
        int level,
        int enableEdge,
        int enableBlur,
        int enableInvert,
        int enableThreshold,
        int thresholdValue,
        void* midU16,
        size_t midU16Bytes,
        void* midF32,
        size_t midF32Bytes,
        void* midI32,
        size_t midI32Bytes);

    IPC_API int32_t __cdecl IPC_UploadRaw16(const void* src, int32_t srcBytes);
    IPC_API int32_t __cdecl IPC_UploadRaw16ToBuffer(const void* src, int32_t srcBytes, int32_t width, int32_t height);
    
    IPC_API int __cdecl CudaReleaseCache();

    IPC_API void* __cdecl IPC_CreateIoSharedHandle(int32_t gpuId, int32_t width, int32_t height);
    IPC_API void   __cdecl IPC_DestroyIoSharedHandle(void* sharedHandle);

    IPC_API void* __cdecl IPC_CreateIoBuffer(int32_t gpuId, int32_t width, int32_t height);
    IPC_API void __cdecl IPC_ReleaseD3D11Resource(void* d3d11Resource);

    IPC_API int32_t __cdecl IPC_GetLastHr();
    IPC_API const char* __cdecl IPC_GetLastErr();
}


extern "C" __declspec(dllexport) int __cdecl CudaAllocDeviceBuffer(size_t bytes, void** outDevPtr);
extern "C" __declspec(dllexport) int __cdecl CudaFreeDeviceBuffer(void* devPtr);
extern "C" __declspec(dllexport) int __cdecl CudaEnsureDeviceBuffer(void** ioDevPtr, size_t* ioBytes, size_t requiredBytes);