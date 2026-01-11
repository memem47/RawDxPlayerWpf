#include <d3d11.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

extern "C" int CudaSetDeviceSafe(int gpuId)
{
    cudaError_t e = cudaSetDevice(gpuId);
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" int CudaRegisterD3D11Texture(void* tex2D, cudaGraphicsResource** outRes)
{
    auto* t = reinterpret_cast<ID3D11Resource*>(tex2D);
    cudaError_t e = cudaGraphicsD3D11RegisterResource(outRes, t, cudaGraphicsRegisterFlagsNone);
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" int CudaUnregister(cudaGraphicsResource* res)
{
    if (!res) return 0;
    cudaError_t e = cudaGraphicsUnregisterResource(res);
    return (e == cudaSuccess) ? 0 : (int)e;
}

// Map and get arrays (keep mapped!)
extern "C" int CudaMapGetArraysMapped(
    cudaGraphicsResource* inRes,
    cudaGraphicsResource* outRes,
    void** inArray,
    void** outArray)
{
    if (!inRes || !outRes || !inArray || !outArray) return (int)cudaErrorInvalidValue;

    cudaError_t e;

    e = cudaGraphicsMapResources(1, &inRes, 0);
    if (e != cudaSuccess) return (int)e;

    e = cudaGraphicsMapResources(1, &outRes, 0);
    if (e != cudaSuccess) {
        cudaGraphicsUnmapResources(1, &inRes, 0);
        return (int)e;
    }

    cudaArray_t inArr = nullptr;
    cudaArray_t outArr = nullptr;

    e = cudaGraphicsSubResourceGetMappedArray(&inArr, inRes, 0, 0);
    if (e != cudaSuccess) return (int)e;

    e = cudaGraphicsSubResourceGetMappedArray(&outArr, outRes, 0, 0);
    if (e != cudaSuccess) return (int)e;

    *inArray = (void*)inArr;
    *outArray = (void*)outArr;
    return 0;
}

extern "C" int CudaUnmapResources(cudaGraphicsResource* inRes, cudaGraphicsResource* outRes)
{
    cudaError_t e;

    e = cudaGraphicsUnmapResources(1, &outRes, 0);
    if (e != cudaSuccess) return (int)e;

    e = cudaGraphicsUnmapResources(1, &inRes, 0);
    if (e != cudaSuccess) return (int)e;

    return 0;
}

// ----------------- kernels -----------------

__device__ __forceinline__ unsigned char wlww_to_u8(unsigned short v, int window, int level)
{
    if (window < 1) window = 1;
    int minv = level - window / 2;
    int maxv = level + window / 2;
    if (maxv <= minv) maxv = minv + 1;

    int iv = (int)v;
    if (iv < minv) iv = minv;
    if (iv > maxv) iv = maxv;

    int out = (int)((iv - minv) * 255.0f / (float)(maxv - minv));
    if (out < 0) out = 0;
    if (out > 255) out = 255;
    return (unsigned char)out;
}

__global__ void WLWWKernel(cudaTextureObject_t texIn16, cudaSurfaceObject_t surfOutBGRA,
    int w, int h, int window, int level)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    // For unnormalized coords, sample at x+0.5, y+0.5
    unsigned short v16 = tex2D<unsigned short>(texIn16, x + 0.5f, y + 0.5f);
    unsigned char g = wlww_to_u8(v16, window, level);

    uchar4 out;
    out.x = g;   // B
    out.y = g;   // G
    out.z = g;   // R
    out.w = 255; // A

    surf2Dwrite(out, surfOutBGRA, x * (int)sizeof(uchar4), y);
}

__global__ void Sobel16Kernel(cudaTextureObject_t texIn16, cudaSurfaceObject_t surfOutBGRA,
    int w, int h, int window, int level)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    auto clampi = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };

    int xm1 = clampi(x - 1, 0, w - 1);
    int xp1 = clampi(x + 1, 0, w - 1);
    int ym1 = clampi(y - 1, 0, h - 1);
    int yp1 = clampi(y + 1, 0, h - 1);

    // read and WL/WW normalize to 0..255 before Sobel
    unsigned char p00 = wlww_to_u8(tex2D<unsigned short>(texIn16, xm1 + 0.5f, ym1 + 0.5f), window, level);
    unsigned char p10 = wlww_to_u8(tex2D<unsigned short>(texIn16, x + 0.5f, ym1 + 0.5f), window, level);
    unsigned char p20 = wlww_to_u8(tex2D<unsigned short>(texIn16, xp1 + 0.5f, ym1 + 0.5f), window, level);

    unsigned char p01 = wlww_to_u8(tex2D<unsigned short>(texIn16, xm1 + 0.5f, y + 0.5f), window, level);
    unsigned char p21 = wlww_to_u8(tex2D<unsigned short>(texIn16, xp1 + 0.5f, y + 0.5f), window, level);

    unsigned char p02 = wlww_to_u8(tex2D<unsigned short>(texIn16, xm1 + 0.5f, yp1 + 0.5f), window, level);
    unsigned char p12 = wlww_to_u8(tex2D<unsigned short>(texIn16, x + 0.5f, yp1 + 0.5f), window, level);
    unsigned char p22 = wlww_to_u8(tex2D<unsigned short>(texIn16, xp1 + 0.5f, yp1 + 0.5f), window, level);

    int g00 = (int)p00, g10 = (int)p10, g20 = (int)p20;
    int g01 = (int)p01, g21 = (int)p21;
    int g02 = (int)p02, g12 = (int)p12, g22 = (int)p22;

    int gx = (-g00 + g20) + (-2 * g01 + 2 * g21) + (-g02 + g22);
    int gy = (-g00 - 2 * g10 - g20) + (g02 + 2 * g12 + g22);

    int mag = abs(gx) + abs(gy);
    if (mag > 255) mag = 255;

    uchar4 out;
    out.x = (unsigned char)mag;
    out.y = (unsigned char)mag;
    out.z = (unsigned char)mag;
    out.w = 255;

    surf2Dwrite(out, surfOutBGRA, x * (int)sizeof(uchar4), y);
}

// ----------------- processing entry -----------------

extern "C" int CudaProcessArrays_R16_To_BGRA(
    void* inArrayVoid,
    void* outArrayVoid,
    int w,
    int h,
    int window,
    int level,
    int enableEdge)
{
    auto inArr = (cudaArray_t)inArrayVoid;
    auto outArr = (cudaArray_t)outArrayVoid;
    if (!inArr || !outArr) return (int)cudaErrorInvalidValue;

    // texture for input R16
    cudaResourceDesc resIn{};
    resIn.resType = cudaResourceTypeArray;
    resIn.res.array.array = inArr;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texIn16 = 0;
    cudaError_t e = cudaCreateTextureObject(&texIn16, &resIn, &texDesc, nullptr);
    if (e != cudaSuccess) return (int)e;

    // surface for output BGRA
    cudaResourceDesc resOut{};
    resOut.resType = cudaResourceTypeArray;
    resOut.res.array.array = outArr;

    cudaSurfaceObject_t surfOut = 0;
    e = cudaCreateSurfaceObject(&surfOut, &resOut);
    if (e != cudaSuccess) {
        cudaDestroyTextureObject(texIn16);
        return (int)e;
    }

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    if (enableEdge)
        Sobel16Kernel << <grid, block >> > (texIn16, surfOut, w, h, window, level);
    else
        WLWWKernel << <grid, block >> > (texIn16, surfOut, w, h, window, level);

    e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();

    cudaDestroySurfaceObject(surfOut);
    cudaDestroyTextureObject(texIn16);

    return (e == cudaSuccess) ? 0 : (int)e;
}
