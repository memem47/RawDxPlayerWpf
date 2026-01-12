#define IMAGEPROCCUDADLL_EXPORTS
#include "ImageProcApi.h"

#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <atomic>
#include <mutex>

using Microsoft::WRL::ComPtr;

static cudaGraphicsResource* g_cudaIO = nullptr;

static ComPtr<ID3D11Device>        g_device;
static ComPtr<ID3D11DeviceContext> g_context;
static ComPtr<ID3D11Texture2D>     g_ioTex;

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

    int32_t __cdecl IPC_Init(int32_t gpuId, void* ioSharedHandle)
    {
        if (!ioSharedHandle) return IPC_ERR_INVALIDARG;

        IPC_Shutdown();

        HRESULT hr = CreateDeviceOnAdapterIndex((int)gpuId, g_device.GetAddressOf(), g_context.GetAddressOf());
        if (FAILED(hr)) return -1;

        hr = g_device->OpenSharedResource((HANDLE)ioSharedHandle, __uuidof(ID3D11Texture2D), (void**)g_ioTex.GetAddressOf());
        if (FAILED(hr)) return -2;

        D3D11_TEXTURE2D_DESC desc{};
        g_ioTex->GetDesc(&desc);

        if (desc.Format != DXGI_FORMAT_R16_UINT) return -5;

        g_w = (int)desc.Width;
        g_h = (int)desc.Height;

        int cr = CudaSetDeviceSafe((int)gpuId);
        if (cr != 0) return -1000 - cr;

        cr = CudaRegisterD3D11Texture(g_ioTex.Get(), &g_cudaIO);
        if (cr != 0) return -1100 - cr;

        // default params
        IPC_Params p{};
        p.sizeBytes = sizeof(IPC_Params);
        p.version = 1;
        p.window = 4000;
        p.level = 2000;
        p.enableEdge = 0;
        p.reserved[0] = 0; // enablePostFilter default OFF
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            g_params = p;
        }

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
        if (!g_cudaIO) return IPC_ERR_NOT_INITIALIZED;

        void* ioArr = nullptr;
        int cr = CudaMapGetArrayMapped(g_cudaIO, &ioArr);
        if (cr != 0) return -1300 - cr;

        IPC_Params p;
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            p = g_params;
        }

        int enablePostFilter = p.reserved[0];

        cr = CudaProcessArray_R16_Inplace(
            ioArr, g_w, g_h,
            p.window, p.level,
            p.enableEdge,
            enablePostFilter);

        int cr2 = CudaUnmapResource(g_cudaIO);

        if (cr != 0) return -1400 - cr;
        if (cr2 != 0) return -1500 - cr2;
        return IPC_OK;
    }

    int32_t __cdecl IPC_Shutdown()
    {
        g_initialized.store(false);

        if (g_cudaIO) { CudaUnregister(g_cudaIO); g_cudaIO = nullptr; }

        CudaReleaseCache();

        g_ioTex.Reset();
        g_context.Reset();
        g_device.Reset();

        g_w = 0;
        g_h = 0;

        return IPC_OK;
    }

} // extern "C"
