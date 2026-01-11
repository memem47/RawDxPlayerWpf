#define IMAGEPROCCUDADLL_EXPORTS
#include "ImageProcApi.h"
#include <atomic>
#include <mutex>

#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

static ComPtr<ID3D11Device>        g_device;
static ComPtr<ID3D11DeviceContext> g_context;
static ComPtr<ID3D11Texture2D>     g_inputTex;
static ComPtr<ID3D11Texture2D>     g_outputTex;


static std::atomic<bool> g_initialized{ false };
static IPC_Params g_params{};
static std::mutex g_mtx;

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

        return 0;
    }

    int __cdecl IPC_SetParams(const IPC_Params* p)
    {
        if (!p || p->sizeBytes < sizeof(IPC_Params))
            return -200;

        g_params = *p;
        return 0;
    }

    int __cdecl IPC_Execute()
    {
        if (!g_context || !g_inputTex || !g_outputTex)
            return -10;

        // GPU 上で input → output コピー
        g_context->CopyResource(
            g_outputTex.Get(),
            g_inputTex.Get()
        );

        g_context->Flush(); // ★追加

        return 0;
    }

    int __cdecl IPC_Shutdown()
    {
        g_inputTex.Reset();
        g_outputTex.Reset();
        g_context.Reset();
        g_device.Reset();
        return 0;
    }

} // extern "C"