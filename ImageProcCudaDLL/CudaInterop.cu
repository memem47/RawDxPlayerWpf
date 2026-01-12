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

__device__ __forceinline__ unsigned short wlww_to_u16(unsigned short v, int window, int level)
{
    if (window < 1) window = 1;

    float minv = (float)level - (float)window * 0.5f;
    float maxv = (float)level + (float)window * 0.5f;
    if (maxv <= minv) maxv = minv + 1.0f;

    float fv = (float)v;
    float u = (fv - minv) / (maxv - minv); // 0..1
    u = fminf(1.0f, fmaxf(0.0f, u));

    float out = u * 65535.0f;
    out = fminf(65535.0f, fmaxf(0.0f, out));
    return (unsigned short)(out + 0.5f);
}

__global__ void WLWW16Kernel(cudaTextureObject_t texIn16, cudaSurfaceObject_t surfOut16,
    int w, int h, int window, int level)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    unsigned short v16 = tex2D<unsigned short>(texIn16, x + 0.5f, y + 0.5f);
    unsigned short out16 = wlww_to_u16(v16, window, level);

    surf2Dwrite(out16, surfOut16, x * (int)sizeof(unsigned short), y);
}

__global__ void Sobel16Kernel(cudaTextureObject_t texIn16, cudaSurfaceObject_t surfOut16,
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

    unsigned short p00 = wlww_to_u16(tex2D<unsigned short>(texIn16, xm1 + 0.5f, ym1 + 0.5f), window, level);
    unsigned short p10 = wlww_to_u16(tex2D<unsigned short>(texIn16, x + 0.5f, ym1 + 0.5f), window, level);
    unsigned short p20 = wlww_to_u16(tex2D<unsigned short>(texIn16, xp1 + 0.5f, ym1 + 0.5f), window, level);

    unsigned short p01 = wlww_to_u16(tex2D<unsigned short>(texIn16, xm1 + 0.5f, y + 0.5f), window, level);
    unsigned short p21 = wlww_to_u16(tex2D<unsigned short>(texIn16, xp1 + 0.5f, y + 0.5f), window, level);

    unsigned short p02 = wlww_to_u16(tex2D<unsigned short>(texIn16, xm1 + 0.5f, yp1 + 0.5f), window, level);
    unsigned short p12 = wlww_to_u16(tex2D<unsigned short>(texIn16, x + 0.5f, yp1 + 0.5f), window, level);
    unsigned short p22 = wlww_to_u16(tex2D<unsigned short>(texIn16, xp1 + 0.5f, yp1 + 0.5f), window, level);

    int gx = -(int)p00 - 2 * (int)p01 - (int)p02 + (int)p20 + 2 * (int)p21 + (int)p22;
    int gy = -(int)p00 - 2 * (int)p10 - (int)p20 + (int)p02 + 2 * (int)p12 + (int)p22;

    float mag = sqrtf((float)(gx * gx + gy * gy));
    mag = fminf(65535.0f, fmaxf(0.0f, mag));

    unsigned short out16 = (unsigned short)(mag + 0.5f);
    surf2Dwrite(out16, surfOut16, x * (int)sizeof(unsigned short), y);
}

// ----------------- processing entry -----------------

extern "C" int CudaProcessArrays_R16_To_R16(
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

    // surface for output R16
    cudaResourceDesc resOut{};
    resOut.resType = cudaResourceTypeArray;
    resOut.res.array.array = outArr;

    cudaSurfaceObject_t surfOut16 = 0;
    e = cudaCreateSurfaceObject(&surfOut16, &resOut);
    if (e != cudaSuccess) {
        cudaDestroyTextureObject(texIn16);
        return (int)e;
    }

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    if (enableEdge)
        Sobel16Kernel << <grid, block >> > (texIn16, surfOut16, w, h, window, level);
    else
        WLWW16Kernel << <grid, block >> > (texIn16, surfOut16, w, h, window, level);

    e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();

    cudaDestroySurfaceObject(surfOut16);
    cudaDestroyTextureObject(texIn16);

    return (e == cudaSuccess) ? 0 : (int)e;
}
