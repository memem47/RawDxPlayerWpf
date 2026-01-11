#define IMAGEPROCCUDADLL_EXPORTS
#include "ImageProcApi.h"

#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <atomic>
#include <mutex>

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

using Microsoft::WRL::ComPtr;

static cudaGraphicsResource* g_cudaIn = nullptr;
static cudaGraphicsResource* g_cudaOut = nullptr;

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
        D3D_DRIVER_TYPE_UNKNOWN,
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

    int32_t __cdecl IPC_Init(int32_t gpuId, void* inSharedHandle, void* outSharedHandle)
    {
        if (!inSharedHandle || !outSharedHandle) return IPC_ERR_INVALIDARG;

        // reset
        IPC_Shutdown();

        HRESULT hr = CreateDeviceOnAdapterIndex((int)gpuId, &g_device, &g_context);
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

        // Expect: input=R16_UINT, output=BGRA8
        if (inDesc.Format != DXGI_FORMAT_R16_UINT) return -5;
        if (outDesc.Format != DXGI_FORMAT_B8G8R8A8_UNORM) return -6;

        g_w = (int)inDesc.Width;
        g_h = (int)inDesc.Height;

        // CUDA device select
        int cr = CudaSetDeviceSafe((int)gpuId);
        if (cr != 0) return -1000 - cr;

        // Register D3D11 textures to CUDA
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

        // Lightweight copy only (safe for frequent GUI updates)
        g_params = *p;
        return IPC_OK;
    }

    int32_t __cdecl IPC_Execute()
    {
        if (!g_initialized.load()) return IPC_ERR_NOT_INITIALIZED;
        if (!g_cudaIn || !g_cudaOut) return IPC_ERR_NOT_INITIALIZED;

        void* inArr = nullptr;
        void* outArr = nullptr;

        // Map and get arrays (keep mapped until after kernel)
        int cr = CudaMapGetArraysMapped(g_cudaIn, g_cudaOut, &inArr, &outArr);
        if (cr != 0) return -1300 - cr;

        // Copy params locally (avoid race during execution)
        IPC_Params p = g_params;

        // Run kernel
        cr = CudaProcessArrays_R16_To_BGRA(inArr, outArr, g_w, g_h, p.window, p.level, p.enableEdge);

        // Unmap resources 반드시実行
        int cr2 = CudaUnmapResources(g_cudaIn, g_cudaOut);
        if (cr != 0) return -1400 - cr;
        if (cr2 != 0) return -1500 - cr2;

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

        return IPC_OK;
    }

} // extern "C"
