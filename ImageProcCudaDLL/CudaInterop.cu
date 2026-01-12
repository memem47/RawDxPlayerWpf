#include <d3d11.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

// ------------------------------------------------------------
// Basic interop helpers
// ------------------------------------------------------------

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

// ------------------------------------------------------------
// Kernels (stage 1)
// ------------------------------------------------------------

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

// ------------------------------------------------------------
// Kernel (stage 2) - Demo post filter: 3x3 box blur
// ------------------------------------------------------------

__global__ void BoxBlur3x3_U16(cudaTextureObject_t texMid16, cudaSurfaceObject_t surfOut16, int w, int h)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    // clamp sampling
    auto sample = [&](int sx, int sy) -> unsigned int {
        sx = sx < 0 ? 0 : (sx >= w ? w - 1 : sx);
        sy = sy < 0 ? 0 : (sy >= h ? h - 1 : sy);
        return (unsigned int)tex2D<unsigned short>(texMid16, sx + 0.5f, sy + 0.5f);
        };

    unsigned int sum = 0;
    sum += sample(x - 1, y - 1); sum += sample(x, y - 1); sum += sample(x + 1, y - 1);
    sum += sample(x - 1, y);     sum += sample(x, y);     sum += sample(x + 1, y);
    sum += sample(x - 1, y + 1); sum += sample(x, y + 1); sum += sample(x + 1, y + 1);

    unsigned short out16 = (unsigned short)((sum + 4) / 9); // round
    surf2Dwrite(out16, surfOut16, x * (int)sizeof(unsigned short), y);
}

// If post filter disabled, just copy mid -> out (2nd stage still keeps pipeline consistent)
__global__ void CopyU16(cudaTextureObject_t texMid16, cudaSurfaceObject_t surfOut16, int w, int h)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    unsigned short v = tex2D<unsigned short>(texMid16, x + 0.5f, y + 0.5f);
    surf2Dwrite(v, surfOut16, x * (int)sizeof(unsigned short), y);
}

// ------------------------------------------------------------
// Processing entry: in(R16) -> [stage1] -> mid(R16) -> [stage2] -> out(R16)
// ------------------------------------------------------------

static cudaError_t CreateTexFromArray(cudaArray_t arr, cudaTextureObject_t* outTex)
{
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = arr;

    cudaTextureDesc tex{};
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModePoint;
    tex.readMode = cudaReadModeElementType;
    tex.normalizedCoords = 0;

    return cudaCreateTextureObject(outTex, &res, &tex, nullptr);
}

static cudaError_t CreateSurfFromArray(cudaArray_t arr, cudaSurfaceObject_t* outSurf)
{
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = arr;
    return cudaCreateSurfaceObject(outSurf, &res);
}

extern "C" int CudaProcessArrays_R16_To_R16(
    void* inArrayVoid,
    void* outArrayVoid,
    int w,
    int h,
    int window,
    int level,
    int enableEdge,
    int enablePostFilter)
{
    auto inArr = (cudaArray_t)inArrayVoid;
    auto outArr = (cudaArray_t)outArrayVoid;
    if (!inArr || !outArr) return (int)cudaErrorInvalidValue;

    cudaError_t e;

    // --- Create input texture (from inArr) ---
    cudaTextureObject_t texIn16 = 0;
    e = CreateTexFromArray(inArr, &texIn16);
    if (e != cudaSuccess) return (int)e;

    // --- Allocate intermediate array (R16 + surface load/store) ---
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<unsigned short>();
    cudaArray_t midArr = nullptr;
    e = cudaMallocArray(&midArr, &ch, (size_t)w, (size_t)h, cudaArraySurfaceLoadStore);
    if (e != cudaSuccess) {
        cudaDestroyTextureObject(texIn16);
        return (int)e;
    }

    // --- Stage1 output surface: mid ---
    cudaSurfaceObject_t surfMid16 = 0;
    e = CreateSurfFromArray(midArr, &surfMid16);
    if (e != cudaSuccess) {
        cudaFreeArray(midArr);
        cudaDestroyTextureObject(texIn16);
        return (int)e;
    }

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    // Stage1: input -> mid
    if (enableEdge)
        Sobel16Kernel << <grid, block >> > (texIn16, surfMid16, w, h, window, level);
    else
        WLWW16Kernel << <grid, block >> > (texIn16, surfMid16, w, h, window, level);

    e = cudaGetLastError();
    if (e != cudaSuccess) {
        cudaDestroySurfaceObject(surfMid16);
        cudaFreeArray(midArr);
        cudaDestroyTextureObject(texIn16);
        return (int)e;
    }

    // --- Create mid texture (read stage2) ---
    cudaTextureObject_t texMid16 = 0;
    e = CreateTexFromArray(midArr, &texMid16);
    if (e != cudaSuccess) {
        cudaDestroySurfaceObject(surfMid16);
        cudaFreeArray(midArr);
        cudaDestroyTextureObject(texIn16);
        return (int)e;
    }

    // --- Create output surface (from outArr) ---
    cudaSurfaceObject_t surfOut16 = 0;
    e = CreateSurfFromArray(outArr, &surfOut16);
    if (e != cudaSuccess) {
        cudaDestroyTextureObject(texMid16);
        cudaDestroySurfaceObject(surfMid16);
        cudaFreeArray(midArr);
        cudaDestroyTextureObject(texIn16);
        return (int)e;
    }

    // Stage2: mid -> out (post filter)
    if (enablePostFilter)
        BoxBlur3x3_U16 << <grid, block >> > (texMid16, surfOut16, w, h);
    else
        CopyU16 << <grid, block >> > (texMid16, surfOut16, w, h);

    e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();

    // cleanup
    cudaDestroySurfaceObject(surfOut16);
    cudaDestroyTextureObject(texMid16);

    cudaDestroySurfaceObject(surfMid16);
    cudaFreeArray(midArr);

    cudaDestroyTextureObject(texIn16);

    return (e == cudaSuccess) ? 0 : (int)e;
}
