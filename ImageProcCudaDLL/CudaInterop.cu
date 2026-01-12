#include <d3d11.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

// ============================================================
// Cached intermediate array (MID) : reuse across frames
// ============================================================

static cudaArray_t g_midArr = nullptr;
static int g_midW = 0;
static int g_midH = 0;

static cudaError_t EnsureMidArrayR16(int w, int h)
{
    if (g_midArr && g_midW == w && g_midH == h) return cudaSuccess;

    if (g_midArr)
    {
        cudaError_t e0 = cudaFreeArray(g_midArr);
        g_midArr = nullptr;
        g_midW = g_midH = 0;
        if (e0 != cudaSuccess) return e0;
    }

    cudaChannelFormatDesc ch = cudaCreateChannelDesc<unsigned short>();
    cudaError_t e = cudaMallocArray(&g_midArr, &ch, (size_t)w, (size_t)h, cudaArraySurfaceLoadStore);
    if (e != cudaSuccess)
    {
        g_midArr = nullptr;
        g_midW = g_midH = 0;
        return e;
    }

    g_midW = w;
    g_midH = h;
    return cudaSuccess;
}

extern "C" __declspec(dllexport) int __cdecl CudaReleaseCache()
{
    if (g_midArr)
    {
        cudaError_t e = cudaFreeArray(g_midArr);
        g_midArr = nullptr;
        g_midW = g_midH = 0;
        return (e == cudaSuccess) ? 0 : (int)e;
    }
    return 0;
}

// ============================================================
// Basic interop helpers
// ============================================================

extern "C" __declspec(dllexport) int __cdecl CudaSetDeviceSafe(int gpuId)
{
    cudaError_t e = cudaSetDevice(gpuId);
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" __declspec(dllexport) int __cdecl CudaRegisterD3D11Texture(void* tex2D, cudaGraphicsResource** outRes)
{
    auto* t = reinterpret_cast<ID3D11Resource*>(tex2D);
    cudaError_t e = cudaGraphicsD3D11RegisterResource(outRes, t, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" __declspec(dllexport) int __cdecl CudaUnregister(cudaGraphicsResource* res)
{
    if (!res) return 0;
    cudaError_t e = cudaGraphicsUnregisterResource(res);
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" __declspec(dllexport) int __cdecl CudaMapGetArrayMapped(cudaGraphicsResource* ioRes, void** ioArray)
{
    if (!ioRes || !ioArray) return (int)cudaErrorInvalidValue;

    cudaError_t e = cudaGraphicsMapResources(1, &ioRes, 0);
    if (e != cudaSuccess) return (int)e;

    cudaArray_t arr = nullptr;
    e = cudaGraphicsSubResourceGetMappedArray(&arr, ioRes, 0, 0);
    if (e != cudaSuccess) return (int)e;

    *ioArray = (void*)arr;
    return 0;
}

extern "C" __declspec(dllexport) int __cdecl CudaUnmapResource(cudaGraphicsResource* ioRes)
{
    cudaError_t e = cudaGraphicsUnmapResources(1, &ioRes, 0);
    return (e == cudaSuccess) ? 0 : (int)e;
}

// ============================================================
// Utility: create texture/surface from array
// ============================================================

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

// ============================================================
// Kernels
// ============================================================

__device__ __forceinline__ unsigned short wlww_to_u16(unsigned short v, int window, int level)
{
    if (window < 1) window = 1;

    float minv = (float)level - (float)window * 0.5f;
    float maxv = (float)level + (float)window * 0.5f;
    if (maxv <= minv) maxv = minv + 1.0f;

    float fv = (float)v;
    float u = (fv - minv) / (maxv - minv);
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

__global__ void BoxBlur3x3_U16(cudaTextureObject_t texIn16, cudaSurfaceObject_t surfOut16, int w, int h)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    auto sample = [&](int sx, int sy) -> unsigned int {
        sx = sx < 0 ? 0 : (sx >= w ? w - 1 : sx);
        sy = sy < 0 ? 0 : (sy >= h ? h - 1 : sy);
        return (unsigned int)tex2D<unsigned short>(texIn16, sx + 0.5f, sy + 0.5f);
        };

    unsigned int sum = 0;
    sum += sample(x - 1, y - 1); sum += sample(x, y - 1); sum += sample(x + 1, y - 1);
    sum += sample(x - 1, y);     sum += sample(x, y);     sum += sample(x + 1, y);
    sum += sample(x - 1, y + 1); sum += sample(x, y + 1); sum += sample(x + 1, y + 1);

    unsigned short out16 = (unsigned short)((sum + 4) / 9);
    surf2Dwrite(out16, surfOut16, x * (int)sizeof(unsigned short), y);
}

__global__ void InvertU16(cudaTextureObject_t texIn16, cudaSurfaceObject_t surfOut16, int w, int h)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    unsigned short v = tex2D<unsigned short>(texIn16, x + 0.5f, y + 0.5f);
    unsigned short out = (unsigned short)(65535u - (unsigned int)v);
    surf2Dwrite(out, surfOut16, x * (int)sizeof(unsigned short), y);
}

__global__ void ThresholdU16(cudaTextureObject_t texIn16, cudaSurfaceObject_t surfOut16,
    int w, int h, unsigned short thresh)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    unsigned short v = tex2D<unsigned short>(texIn16, x + 0.5f, y + 0.5f);
    unsigned short out = (v >= thresh) ? 65535 : 0;
    surf2Dwrite(out, surfOut16, x * (int)sizeof(unsigned short), y);
}

__global__ void CopyU16(cudaTextureObject_t texIn16, cudaSurfaceObject_t surfOut16, int w, int h)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    unsigned short v = tex2D<unsigned short>(texIn16, x + 0.5f, y + 0.5f);
    surf2Dwrite(v, surfOut16, x * (int)sizeof(unsigned short), y);
}

// ============================================================
// In-place processing entry (IO <-> MID), with cached MID
// ============================================================

extern "C" __declspec(dllexport) int __cdecl CudaProcessArray_R16_Inplace(
    void* ioArrayVoid,
    int w,
    int h,
    int window,
    int level,
    int enableEdge,
    int enableBlur,
    int enableInvert,
    int enableThreshold,
    int thresholdValue)
{
    cudaArray_t ioArr = (cudaArray_t)ioArrayVoid;
    if (!ioArr) return (int)cudaErrorInvalidValue;

    cudaError_t e = EnsureMidArrayR16(w, h);
    if (e != cudaSuccess) return (int)e;

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    // curIsIO==true => latest result in ioArr, false => in g_midArr
    bool curIsIO = true;

    auto SrcArr = [&]() -> cudaArray_t { return curIsIO ? ioArr : g_midArr; };
    auto DstArr = [&]() -> cudaArray_t { return curIsIO ? g_midArr : ioArr; };

    auto RunOp = [&](auto launcher) -> cudaError_t
        {
            cudaTextureObject_t tex = 0;
            cudaSurfaceObject_t surf = 0;

            cudaArray_t src = SrcArr();
            cudaArray_t dst = DstArr();

            cudaError_t ee = CreateTexFromArray(src, &tex);
            if (ee != cudaSuccess) return ee;

            ee = CreateSurfFromArray(dst, &surf);
            if (ee != cudaSuccess) { cudaDestroyTextureObject(tex); return ee; }

            launcher(tex, surf);

            ee = cudaGetLastError();

            cudaDestroySurfaceObject(surf);
            cudaDestroyTextureObject(tex);

            if (ee != cudaSuccess) return ee;

            curIsIO = !curIsIO; // swap
            return cudaSuccess;
        };

    // Stage1: always execute (WLWW or Sobel)
    e = RunOp([&](cudaTextureObject_t tex, cudaSurfaceObject_t surf) {
        if (enableEdge)
            Sobel16Kernel << <grid, block >> > (tex, surf, w, h, window, level);
        else
            WLWW16Kernel << <grid, block >> > (tex, surf, w, h, window, level);
        });
    if (e != cudaSuccess) return (int)e;

    // Stage2: blur
    if (enableBlur)
    {
        e = RunOp([&](cudaTextureObject_t tex, cudaSurfaceObject_t surf) {
            BoxBlur3x3_U16 << <grid, block >> > (tex, surf, w, h);
            });
        if (e != cudaSuccess) return (int)e;
    }

    // Stage3: invert
    if (enableInvert)
    {
        e = RunOp([&](cudaTextureObject_t tex, cudaSurfaceObject_t surf) {
            InvertU16 << <grid, block >> > (tex, surf, w, h);
            });
        if (e != cudaSuccess) return (int)e;
    }

    // Stage4: threshold
    if (enableThreshold)
    {
        int tv = thresholdValue;
        if (tv < 0) tv = 0;
        if (tv > 65535) tv = 65535;
        unsigned short th = (unsigned short)tv;

        e = RunOp([&](cudaTextureObject_t tex, cudaSurfaceObject_t surf) {
            ThresholdU16 << <grid, block >> > (tex, surf, w, h, th);
            });
        if (e != cudaSuccess) return (int)e;
    }

    // Ensure final result is in IO
    if (!curIsIO)
    {
        e = RunOp([&](cudaTextureObject_t tex, cudaSurfaceObject_t surf) {
            CopyU16 << <grid, block >> > (tex, surf, w, h);
            });
        if (e != cudaSuccess) return (int)e;
        // now curIsIO should be true
    }

    e = cudaDeviceSynchronize();
    return (e == cudaSuccess) ? 0 : (int)e;
}
