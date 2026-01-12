#define IMAGEPROCCUDADLL_EXPORTS
#include "ImageProcApi.h"

#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <atomic>
#include <mutex>

using Microsoft::WRL::ComPtr;

// CUDA headers only in implementation
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

static cudaGraphicsResource* g_cudaIn = nullptr;
static cudaGraphicsResource* g_cudaOut = nullptr;

static ComPtr<ID3D11Device>        g_device;
static ComPtr<ID3D11DeviceContext> g_context;
static ComPtr<ID3D11Texture2D>     g_inputTex;
static ComPtr<ID3D11Texture2D>     g_outputTex;

static std::atomic<bool> g_initialized{ false };
static IPC_Params g_params{};
static std::mutex g_mtx;

static int g_w = 0;
static int g_h = 0;

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

    static const D3D_FEATURE_LEVEL levels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0
    };
    D3D_FEATURE_LEVEL outLevel = D3D_FEATURE_LEVEL_11_0;

    hr = D3D11CreateDevice(
        adapter.Get(),
        D3D_DRIVER_TYPE_UNKNOWN,
        nullptr,
        flags,
        levels,
        (UINT)_countof(levels),
        D3D11_SDK_VERSION,
        dev,
        &outLevel,
        ctx);

    return hr;
}

extern "C" {

    int32_t __cdecl IPC_Init(int32_t gpuId, void* inSharedHandle, void* outSharedHandle)
    {
        if (!inSharedHandle || !outSharedHandle) return IPC_ERR_INVALIDARG;

        // reset
        IPC_Shutdown();

        HRESULT hr = CreateDeviceOnAdapterIndex((int)gpuId, g_device.GetAddressOf(), g_context.GetAddressOf());
        if (FAILED(hr)) return -1;

        hr = g_device->OpenSharedResource((HANDLE)inSharedHandle, __uuidof(ID3D11Texture2D), (void**)g_inputTex.GetAddressOf());
        if (FAILED(hr)) return -2;

        hr = g_device->OpenSharedResource((HANDLE)outSharedHandle, __uuidof(ID3D11Texture2D), (void**)g_outputTex.GetAddressOf());
        if (FAILED(hr)) return -3;

        // Validate formats (IMPORTANT!)
        D3D11_TEXTURE2D_DESC inDesc{}, outDesc{};
        g_inputTex->GetDesc(&inDesc);
        g_outputTex->GetDesc(&outDesc);

        if (inDesc.Width != outDesc.Width || inDesc.Height != outDesc.Height) return -4;

        // Expect: input=R16_UINT, output=R16_UINT
        if (inDesc.Format != DXGI_FORMAT_R16_UINT) return -5;
        if (outDesc.Format != DXGI_FORMAT_R16_UINT) return -6;

        g_w = (int)inDesc.Width;
        g_h = (int)inDesc.Height;

        // CUDA device
        int cr = CudaSetDeviceSafe((int)gpuId);
        if (cr != 0) return -1000 - cr;

        cr = CudaRegisterD3D11Texture(g_inputTex.Get(), &g_cudaIn);
        if (cr != 0) return -1100 - cr;

        cr = CudaRegisterD3D11Texture(g_outputTex.Get(), &g_cudaOut);
        if (cr != 0) return -1200 - cr;

        // default params
        IPC_Params p{};
        p.sizeBytes = sizeof(IPC_Params);
        p.version = 1;
        p.window = 4000;
        p.level = 2000;
        p.enableEdge = 0;
        g_params = p;

        g_initialized.store(true);
        return IPC_OK;
    }

    int32_t __cdecl IPC_SetParams(const IPC_Params* p)
    {
        int32_t v = ValidateParams(p);
        if (v != IPC_OK) return v;
        if (!g_initialized.load()) return IPC_ERR_NOT_INITIALIZED;

        std::lock_guard<std::mutex> lk(g_mtx);
        g_params = *p;
        return IPC_OK;
    }

    int32_t __cdecl IPC_Execute()
    {
        if (!g_initialized.load()) return IPC_ERR_NOT_INITIALIZED;

        IPC_Params p;
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            p = g_params;
        }

        void* inArr = nullptr;
        void* outArr = nullptr;

        int cr = CudaMapGetArraysMapped(g_cudaIn, g_cudaOut, &inArr, &outArr);
        if (cr != 0) return -2000 - cr;

        cr = CudaProcessArrays_R16_To_R16(inArr, outArr, g_w, g_h, p.window, p.level, p.enableEdge);

        int ur = CudaUnmapResources(g_cudaIn, g_cudaOut);
        if (ur != 0) return -2100 - ur;

        if (cr != 0) return -2200 - cr;
        return IPC_OK;
    }

    int32_t __cdecl IPC_Shutdown()
    {
        g_initialized.store(false);

        if (g_cudaOut) { CudaUnregister(g_cudaOut); g_cudaOut = nullptr; }
        if (g_cudaIn) { CudaUnregister(g_cudaIn);  g_cudaIn = nullptr; }

        g_inputTex.Reset();
        g_outputTex.Reset();
        g_context.Reset();
        g_device.Reset();

        g_w = 0;
        g_h = 0;

        return IPC_OK;
    }

} // extern "C"
