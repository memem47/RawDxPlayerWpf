#define IMAGEPROCCUDADLL_EXPORTS
#include "ImageProcApi.h"
#include <atomic>
#include <mutex>

#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

static cudaGraphicsResource* g_cudaIn = nullptr;
static cudaGraphicsResource* g_cudaOut = nullptr;

using Microsoft::WRL::ComPtr;

static ComPtr<ID3D11Device>        g_device;
static ComPtr<ID3D11DeviceContext> g_context;
static ComPtr<ID3D11Texture2D>     g_inputTex;
static ComPtr<ID3D11Texture2D>     g_outputTex;


static std::atomic<bool> g_initialized{ false };
static IPC_Params g_params{};
static std::mutex g_mtx;

static int g_w = 0, g_h = 0;

static int32_t ValidateParams(const IPC_Params* p)
{
    if (!p) return IPC_ERR_INVALIDARG;
    if (p->sizeBytes < sizeof(IPC_Params)) return IPC_ERR_INVALIDARG;
    if (p->version != 1) return IPC_ERR_INVALIDARG;
    return IPC_OK;
}

static HRESULT CreateDeviceOnAdapterIndex(int gpuId, ID3D11Device** dev, ID3D11DeviceContext** ctx)
{
    ComPtr<IDXGIFactory1> factory;
    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)factory.GetAddressOf());
    if (FAILED(hr)) return hr;

    ComPtr<IDXGIAdapter1> adapter;
    hr = factory->EnumAdapters1((UINT)gpuId, adapter.GetAddressOf());
    if (FAILED(hr)) return hr;

    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

    D3D_FEATURE_LEVEL fl;
    return D3D11CreateDevice(
        adapter.Get(),
        D3D_DRIVER_TYPE_UNKNOWN, // ← adapter指定時は UNKNOWN
        nullptr,
        flags,
        nullptr, 0,
        D3D11_SDK_VERSION,
        dev,
        &fl,
        ctx
    );
}
extern "C" {
    int __cdecl IPC_Init(int gpuId, HANDLE inSharedHandle, HANDLE outSharedHandle)
    {
        (void)gpuId;

        if (!inSharedHandle || !outSharedHandle)
            return -100;

        g_inputTex.Reset();
        g_outputTex.Reset();
        g_context.Reset();
        g_device.Reset();

        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

        D3D_FEATURE_LEVEL fl;

        HRESULT hr = CreateDeviceOnAdapterIndex(gpuId, &g_device, &g_context);
        if (FAILED(hr)) return -1;

        hr = g_device->OpenSharedResource(inSharedHandle, __uuidof(ID3D11Texture2D), (void**)g_inputTex.GetAddressOf());
        if (FAILED(hr)) return -2;

        hr = g_device->OpenSharedResource(outSharedHandle, __uuidof(ID3D11Texture2D), (void**)g_outputTex.GetAddressOf());
        if (FAILED(hr)) return -3;

        // 念のためフォーマット・サイズ確認
        D3D11_TEXTURE2D_DESC inDesc{}, outDesc{};
        g_inputTex->GetDesc(&inDesc);
        g_outputTex->GetDesc(&outDesc);

        if (inDesc.Width != outDesc.Width ||
            inDesc.Height != outDesc.Height ||
            inDesc.Format != outDesc.Format)
            return -4;

        // CUDA device 選択
        int cr = CudaSetDeviceSafe(gpuId);
        if (cr != 0) return -1000 - cr;

        // CUDA に D3D11 texture を登録
        cr = CudaRegisterD3D11Texture(g_inputTex.Get(), &g_cudaIn);
        if (cr != 0) return -1100 - cr;

        cr = CudaRegisterD3D11Texture(g_outputTex.Get(), &g_cudaOut);
        if (cr != 0) return -1200 - cr;


        D3D11_TEXTURE2D_DESC desc{};
        g_inputTex->GetDesc(&desc);
        g_w = (int)desc.Width;
        g_h = (int)desc.Height;

        return 0;
    }

    int __cdecl IPC_SetParams(const IPC_Params* p)
    {
        if (!p || p->sizeBytes < sizeof(IPC_Params))
            return -200;

        g_params = *p;
        return 0;
    }

    //int __cdecl IPC_Execute()
    //{
    //    if (!g_context || !g_inputTex || !g_outputTex)
    //        return -10;

    //    // GPU 上で input → output コピー
    //    g_context->CopyResource(
    //        g_outputTex.Get(),
    //        g_inputTex.Get()
    //    );

    //    g_context->Flush(); // ★追加

    //    return 0;
    //}

    int __cdecl IPC_Execute()
    {
        if (!g_cudaIn || !g_cudaOut) return -20;

        void* inArr = nullptr;
        void* outArr = nullptr;

        int cr = CudaMapGetArrays(g_cudaIn, g_cudaOut, &inArr, &outArr);
        if (cr != 0) return -1300 - cr;

        // パラメータ enableEdge を反映
        int enableEdge = 1;//g_params.enableEdge;

        cr = CudaProcessArrays(inArr, outArr, g_w, g_h, enableEdge);
        if (cr != 0) return -1400 - cr;

        return 0;
    }


    int __cdecl IPC_Shutdown()
    {
        CudaUnregister(g_cudaOut); g_cudaOut = nullptr;
        CudaUnregister(g_cudaIn);  g_cudaIn = nullptr;

        // 既存のD3D解放
        g_inputTex.Reset();
        g_outputTex.Reset();
        g_context.Reset();
        g_device.Reset();
        return 0;
    }

} // extern "C"