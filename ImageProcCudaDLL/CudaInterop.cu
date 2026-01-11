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
    // tex2D は ID3D11Texture2D*
    auto* t = reinterpret_cast<ID3D11Resource*>(tex2D);

    // 最小は None。書き込み先は out 側で WriteDiscard を付けてもよいがまずは無しで通す。
    cudaError_t e = cudaGraphicsD3D11RegisterResource(outRes, t, cudaGraphicsRegisterFlagsNone);
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" int CudaUnregister(cudaGraphicsResource* res)
{
    if (!res) return 0;
    cudaError_t e = cudaGraphicsUnregisterResource(res);
    return (e == cudaSuccess) ? 0 : (int)e;
}

// ★宣言と一致：void** にする
extern "C" int CudaMapGetArrays(
    cudaGraphicsResource* inRes,
    cudaGraphicsResource* outRes,
    void** inArray,
    void** outArray)
{
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
    if (e != cudaSuccess) {
        cudaGraphicsUnmapResources(1, &outRes, 0);
        cudaGraphicsUnmapResources(1, &inRes, 0);
        return (int)e;
    }

    e = cudaGraphicsSubResourceGetMappedArray(&outArr, outRes, 0, 0);
    if (e != cudaSuccess) {
        cudaGraphicsUnmapResources(1, &outRes, 0);
        cudaGraphicsUnmapResources(1, &inRes, 0);
        return (int)e;
    }

    // 呼び出し側へ返す（型は void* として返す）
    *inArray = (void*)inArr;
    *outArray = (void*)outArr;

    e = cudaGraphicsUnmapResources(1, &outRes, 0);
    if (e != cudaSuccess) {
        cudaGraphicsUnmapResources(1, &inRes, 0);
        return (int)e;
    }

    e = cudaGraphicsUnmapResources(1, &inRes, 0);
    if (e != cudaSuccess) return (int)e;

    return 0;
}

static __device__ __forceinline__ unsigned char to_gray(uchar4 p)
{
    // BGRA: p.x=B, p.y=G, p.z=R, p.w=A
    int g = (int)p.x + (int)p.y + (int)p.z;
    g /= 3;
    return (unsigned char)g;
}

__global__ void SobelEdgeKernel(cudaTextureObject_t texIn, cudaSurfaceObject_t surfOut, int w, int h)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    // clamp helper
    auto clampi = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };

    int xm1 = clampi(x - 1, 0, w - 1);
    int xp1 = clampi(x + 1, 0, w - 1);
    int ym1 = clampi(y - 1, 0, h - 1);
    int yp1 = clampi(y + 1, 0, h - 1);

    // tex2D uses float coords; with normalizedCoords=0, texel space. Sample at x+0.5
    uchar4 p00 = tex2D<uchar4>(texIn, xm1 + 0.5f, ym1 + 0.5f);
    uchar4 p10 = tex2D<uchar4>(texIn, x + 0.5f, ym1 + 0.5f);
    uchar4 p20 = tex2D<uchar4>(texIn, xp1 + 0.5f, ym1 + 0.5f);

    uchar4 p01 = tex2D<uchar4>(texIn, xm1 + 0.5f, y + 0.5f);
    uchar4 p21 = tex2D<uchar4>(texIn, xp1 + 0.5f, y + 0.5f);

    uchar4 p02 = tex2D<uchar4>(texIn, xm1 + 0.5f, yp1 + 0.5f);
    uchar4 p12 = tex2D<uchar4>(texIn, x + 0.5f, yp1 + 0.5f);
    uchar4 p22 = tex2D<uchar4>(texIn, xp1 + 0.5f, yp1 + 0.5f);

    int g00 = to_gray(p00), g10 = to_gray(p10), g20 = to_gray(p20);
    int g01 = to_gray(p01), g21 = to_gray(p21);
    int g02 = to_gray(p02), g12 = to_gray(p12), g22 = to_gray(p22);

    // Sobel
    int gx = (-g00 + g20) + (-2 * g01 + 2 * g21) + (-g02 + g22);
    int gy = (-g00 - 2 * g10 - g20) + (g02 + 2 * g12 + g22);

    int mag = abs(gx) + abs(gy);
    if (mag > 255) mag = 255;

    uchar4 out;
    out.x = (unsigned char)mag; // B
    out.y = (unsigned char)mag; // G
    out.z = (unsigned char)mag; // R
    out.w = 255;                // A

    // surf2Dwrite expects byte offset for x
    surf2Dwrite(out, surfOut, x * (int)sizeof(uchar4), y);
}

// CUDA array in/out を受け取り、enableEdge に応じて out へ書く
extern "C" int CudaProcessArrays(void* inArrayVoid, void* outArrayVoid, int w, int h, int enableEdge)
{
    auto inArr = (cudaArray_t)inArrayVoid;
    auto outArr = (cudaArray_t)outArrayVoid;

    if (!inArr || !outArr) return (int)cudaErrorInvalidValue;

    if (!enableEdge)
    {
        // in -> out copy (device)
        cudaError_t e = cudaMemcpy2DArrayToArray(
            outArr, 0, 0,
            inArr, 0, 0,
            (size_t)w * sizeof(uchar4),
            (size_t)h,
            cudaMemcpyDeviceToDevice);
        return (e == cudaSuccess) ? 0 : (int)e;
    }

    // texture object for input
    cudaResourceDesc resIn{};
    resIn.resType = cudaResourceTypeArray;
    resIn.res.array.array = inArr;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texIn = 0;
    cudaError_t e = cudaCreateTextureObject(&texIn, &resIn, &texDesc, nullptr);
    if (e != cudaSuccess) return (int)e;

    // surface object for output
    cudaResourceDesc resOut{};
    resOut.resType = cudaResourceTypeArray;
    resOut.res.array.array = outArr;

    cudaSurfaceObject_t surfOut = 0;
    e = cudaCreateSurfaceObject(&surfOut, &resOut);
    if (e != cudaSuccess)
    {
        cudaDestroyTextureObject(texIn);
        return (int)e;
    }

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    SobelEdgeKernel << <grid, block >> > (texIn, surfOut, w, h);
    e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();

    cudaDestroySurfaceObject(surfOut);
    cudaDestroyTextureObject(texIn);

    return (e == cudaSuccess) ? 0 : (int)e;
}
