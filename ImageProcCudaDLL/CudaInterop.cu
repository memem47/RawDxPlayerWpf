#include <d3d11.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

struct CudaContext
{
    int device = 0;
    cudaStream_t stream = nullptr;

    // Optional: keep cache buffers here (replacing static globals)
    void* midBuf = nullptr;       // linear temp buffer
    size_t midBufBytes = 0;

    cudaArray_t midArr = nullptr; // optional, if you use array/surface temp
    int w = 0;
    int h = 0;

    bool inited = false;
};

static void CudaContext_Destroy(CudaContext* c)
{
    if (!c) return;

    if (c->midArr)
    {
        // If allocated with cudaMallocArray / cudaFreeArray
        cudaFreeArray(c->midArr);
        c->midArr = nullptr;
    }

    if (c->midBuf)
    {
        // IMPORTANT: use same stream for freeAsync
        cudaFreeAsync(c->midBuf, c->stream);
        c->midBuf = nullptr;
        c->midBufBytes = 0;
    }

    if (c->stream)
    {
        cudaStreamDestroy(c->stream);
        c->stream = nullptr;
    }

    c->inited = false;
}

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

//------------------------------------------------------------------------------ 
// グローバル中間バッファ（16bit R チャンネル用） 
//------------------------------------------------------------------------------ 
// GPU 上に確保する 16bit 中間バッファ
static unsigned short* g_midBuf = nullptr;
// 現在確保されているバッファの幅・高さ
static int g_midW2 = 0;
static int g_midH2 = 0;

//------------------------------------------------------------------------------
// EnsureMidBufferR16 
// 指定されたサイズ (w x h) の 16bit 中間バッファが GPU 上に存在することを保証する。 
// 既存バッファが同サイズであれば再確保は行わない。 
// サイズが異なる場合は既存バッファを解放し、新たに確保する。 
// 戻り値: 
//   cudaSuccess 正常終了 
//   その他の値 CUDA API からのエラーコード
// ------------------------------------------------------------------------------
static cudaError_t EnsureMidBufferR16(int w, int h)
{
    // 既存バッファが要求サイズと一致していれば何もしない
    if (g_midBuf && g_midW2 == w && g_midH2 == h) return cudaSuccess;

    // サイズ不一致または未確保の場合、既存バッファを解放
    if (g_midBuf)
    {
        cudaError_t e0 = cudaFree(g_midBuf);
        g_midBuf = nullptr;
        g_midW2 = g_midH2 = 0;
        // 解放に失敗した場合は即座にエラーを返す
        if (e0 != cudaSuccess) return e0;
    }

    // 新しいバッファを確保
    size_t bytes = (size_t)w * (size_t)h * sizeof(unsigned short);
    cudaError_t e = cudaMalloc((void**)&g_midBuf, bytes);
    if (e != cudaSuccess)
    {
        // 確保失敗時は状態をクリアしてエラーを返す
        g_midBuf = nullptr;
        g_midW2 = g_midH2 = 0;
        return e;
    }

    // 正常に確保できたのでサイズ情報を更新
    g_midW2 = w;
    g_midH2 = h;

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

    if (g_midBuf)
    {
        cudaError_t e2 = cudaFree(g_midBuf);
        g_midBuf = nullptr;
        g_midW2 = g_midH2 = 0;
        if (e2 != cudaSuccess) return (int)e2;
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


extern "C" __declspec(dllexport) int __cdecl CudaRegisterD3D11Buffer(void* buf, cudaGraphicsResource** outRes)
{
    auto* t = reinterpret_cast<ID3D11Buffer*>(buf);
    cudaError_t e = cudaGraphicsD3D11RegisterResource(outRes, t, cudaGraphicsRegisterFlagsNone);
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" __declspec(dllexport) int __cdecl CudaUnregister(cudaGraphicsResource* res)
{
    if (!res) return 0;
    cudaError_t e = cudaGraphicsUnregisterResource(res);
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" __declspec(dllexport) int __cdecl CudaMapGetArrayMapped(
    cudaGraphicsResource* ioRes, void** ioArray)
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

extern "C" __declspec(dllexport) int __cdecl CudaMapGetPointerMapped(
    cudaGraphicsResource* ioRes, void** devPtr, size_t* outBytes)
{
    if (!ioRes || !devPtr) return (int)cudaErrorInvalidValue;

    cudaError_t e = cudaGraphicsMapResources(1, &ioRes, 0);
    if (e != cudaSuccess) return (int)e;

    void* p = nullptr;
    size_t bytes = 0;
    e = cudaGraphicsResourceGetMappedPointer(&p, &bytes, ioRes);
    if (e != cudaSuccess) return (int)e;

    *devPtr = p;
    if (outBytes) *outBytes = bytes;
    return 0;
}


extern "C" __declspec(dllexport) int __cdecl CudaUnmapResource(cudaGraphicsResource* ioRes)
{
    cudaError_t e = cudaGraphicsUnmapResources(1, &ioRes, 0);
    return (e == cudaSuccess) ? 0 : (int)e;

    cudaArray_t ioArr = nullptr;

    e = cudaGraphicsSubResourceGetMappedArray(&ioArr, ioRes, 0, 0);
    if (e != cudaSuccess) return (int)e;

    return 0;
}

extern "C" int CudaUnmapResources(cudaGraphicsResource* ioRes)
{
    cudaError_t e;

    e = cudaGraphicsUnmapResources(1, &ioRes, 0);
    if (e != cudaSuccess) return (int)e;

    return 0;
}

extern "C" int CudaContext_Init(CudaContext* c, int device)
{
    if (!c) return -1;
    c->device = device;

    cudaError_t e = cudaSetDevice(device);
    if (e != cudaSuccess) return (int)e;

    e = cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking);
    if (e != cudaSuccess) return (int)e;

    // ---- mempool tuning (optional but recommended) ----
    cudaMemPool_t pool = nullptr;
    e = cudaDeviceGetDefaultMemPool(&pool, device);
    if (e == cudaSuccess && pool)
    {
        // keep cached memory (example: 512MB; adjust later)
        unsigned long long threshold = 512ull * 1024ull * 1024ull;
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    }

    c->inited = true;
    return 0;
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

__device__ __forceinline__ int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void WLWW16Kernel_Buf(const unsigned short* in, unsigned short* out,
    int w, int h, int window, int level)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    unsigned short v = in[y * w + x];
    out[y * w + x] = wlww_to_u16(v, window, level);
}

__global__ void Sobel16Kernel_Buf(const unsigned short* in, unsigned short* out,
    int w, int h, int window, int level)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    int xm1 = clampi(x - 1, 0, w - 1);
    int xp1 = clampi(x + 1, 0, w - 1);
    int ym1 = clampi(y - 1, 0, h - 1);
    int yp1 = clampi(y + 1, 0, h - 1);

    auto S = [&](int sx, int sy) -> unsigned short {
        return wlww_to_u16(in[sy * w + sx], window, level);
        };

    unsigned short p00 = S(xm1, ym1);
    unsigned short p10 = S(x, ym1);
    unsigned short p20 = S(xp1, ym1);

    unsigned short p01 = S(xm1, y);
    unsigned short p21 = S(xp1, y);

    unsigned short p02 = S(xm1, yp1);
    unsigned short p12 = S(x, yp1);
    unsigned short p22 = S(xp1, yp1);

    int gx = -(int)p00 - 2 * (int)p01 - (int)p02 + (int)p20 + 2 * (int)p21 + (int)p22;
    int gy = -(int)p00 - 2 * (int)p10 - (int)p20 + (int)p02 + 2 * (int)p12 + (int)p22;

    float mag = sqrtf((float)(gx * gx + gy * gy));
    mag = fminf(65535.0f, fmaxf(0.0f, mag));
    out[y * w + x] = (unsigned short)(mag + 0.5f);
}

__global__ void BoxBlur3x3_Buf(const unsigned short* in, unsigned short* out, int w, int h)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    unsigned int sum = 0;
    for (int dy = -1; dy <= 1; dy++)
    {
        int sy = clampi(y + dy, 0, h - 1);
        for (int dx = -1; dx <= 1; dx++)
        {
            int sx = clampi(x + dx, 0, w - 1);
            sum += (unsigned int)in[sy * w + sx];
        }
    }
    out[y * w + x] = (unsigned short)((sum + 4) / 9);
}

__global__ void InvertU16_Buf(const unsigned short* in, unsigned short* out, int w, int h)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    unsigned short v = in[y * w + x];
    out[y * w + x] = (unsigned short)(65535u - (unsigned int)v);
}

__global__ void ThresholdU16_Buf(const unsigned short* in, unsigned short* out, int w, int h, unsigned short th)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    unsigned short v = in[y * w + x];
    out[y * w + x] = (v >= th) ? 65535 : 0;
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

//------------------------------------------------------------------------------ 
// CudaProcessBuffer_R16_Inplace 
// 16bit グレースケール画像バッファ（GPU メモリ上）に対して、指定された 
// 一連の画像処理（WL/WW、Sobel、Blur、Invert、Threshold）をインプレースで実行する。 
// 引数: 
//   ioDevPtrVoid : GPU 上の入力兼出力バッファ（unsigned short*） 
//   w, h : 画像サイズ（幅・高さ） 
//   window, level : WL/WW または Sobel 用パラメータ 
//   enableEdge : Sobel エッジ検出を有効化（false の場合は WL/WW） 
//   enableBlur : 3x3 ボックスブラーを有効化 
//   enableInvert : 階調反転を有効化
//   enableThreshold: 二値化処理を有効化
//   thresholdValue : 二値化の閾値（0-65535）
// 戻り値: 
//   0 : 正常終了
//  その他 : CUDA エラーコード 
// ------------------------------------------------------------------------------
extern "C" __declspec(dllexport) int __cdecl CudaProcessBuffer_R16_Inplace(
    void* ioDevPtrVoid,
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
    // 入力ポインタと画像サイズの妥当性チェック
    if (!ioDevPtrVoid) return (int)cudaErrorInvalidValue;
    if (w <= 0 || h <= 0) return (int)cudaErrorInvalidValue;

    // 入出力バッファ（GPU メモリ）
    unsigned short* io = (unsigned short*)ioDevPtrVoid;

    // 中間バッファの確保（必要に応じて再確保）
    cudaError_t e = EnsureMidBufferR16(w, h);
    if (e != cudaSuccess) return (int)e;

    // CUDA カーネルの実行構成
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    // curIsIO == true : 最新データが io にある 
    // curIsIO == false: 最新データが g_midBuf にある
    bool curIsIO = true;

    // 入力元と出力先を動的に切り替えるラムダ
    auto Src = [&]() -> const unsigned short* { return curIsIO ? io : g_midBuf; };
    auto Dst = [&]() -> unsigned short* { return curIsIO ? g_midBuf : io; };

    // カーネル実行 → エラーチェック → バッファ切り替え を共通化
    auto RunOp = [&](auto launcher) -> cudaError_t {
        launcher(Src(), Dst());
        cudaError_t ee = cudaGetLastError();
        if (ee != cudaSuccess) return ee;
        curIsIO = !curIsIO;
        return cudaSuccess;
        };

    // Stage1: WLWW or Sobel (always)
    e = RunOp([&](const unsigned short* in, unsigned short* out) {
        if (enableEdge)
            Sobel16Kernel_Buf << <grid, block >> > (in, out, w, h, window, level);
        else
            WLWW16Kernel_Buf << <grid, block >> > (in, out, w, h, window, level);
        });
    if (e != cudaSuccess) return (int)e;

    // Stage2: blur
    if (enableBlur)
    {
        e = RunOp([&](const unsigned short* in, unsigned short* out) {
            BoxBlur3x3_Buf << <grid, block >> > (in, out, w, h);
            });
        if (e != cudaSuccess) return (int)e;
    }

    // Stage3: invert
    if (enableInvert)
    {
        e = RunOp([&](const unsigned short* in, unsigned short* out) {
            InvertU16_Buf << <grid, block >> > (in, out, w, h);
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

        e = RunOp([&](const unsigned short* in, unsigned short* out) {
            ThresholdU16_Buf << <grid, block >> > (in, out, w, h, th);
            });
        if (e != cudaSuccess) return (int)e;
    }

    // 最終結果が io にあることを保証
    if (!curIsIO)
    {
        // 最新データが g_midBuf にあるため io にコピーする
        size_t bytes = (size_t)w * (size_t)h * sizeof(unsigned short);
        e = cudaMemcpy(io, g_midBuf, bytes, cudaMemcpyDeviceToDevice);
        if (e != cudaSuccess) return (int)e;
    }

    // 全カーネルの完了を同期
    e = cudaDeviceSynchronize();
    return (e == cudaSuccess) ? 0 : (int)e;
}
