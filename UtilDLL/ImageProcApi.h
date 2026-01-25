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


    // ----------------------------------------------------------------------
    // Global-less variants (do NOT rely on g_device / g_context / g_ioBuf).
    //
    // - ioBufferPtr: ID3D11Buffer* (created by IPC_CreateIoBuffer etc.)
    // - The function obtains ID3D11Device and immediate context from the buffer.
    // - width/height are required to validate src/dst sizes (bytes = w*h*2).
    // - gpuId is accepted for API symmetry/logging; the actual device is derived
    //   from the buffer (so gpuId is not required to match, but SHOULD match).
    // ----------------------------------------------------------------------
    IPC_API int32_t __cdecl IPC_UploadRaw16ToBufferEx(
        int32_t gpuId, void* ioBufferPtr,
        const void* src, int32_t srcBytes,
        int32_t width, int32_t height);

    IPC_API int32_t __cdecl IPC_ReadbackRaw16FromBufferEx(
        int32_t gpuId, void* ioBufferPtr,
        void* dst, int32_t dstBytes,
        int32_t width, int32_t height);


    IPC_API void* __cdecl IPC_CreateIoBuffer(int32_t gpuId, int32_t width, int32_t height);
}
