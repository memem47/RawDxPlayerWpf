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

static ComPtr<ID3D11Texture2D>     g_stagingReadback;

static std::atomic<bool> g_initialized{ false };
static IPC_Params g_params{};
static std::mutex g_mtx;

static int g_w = 0;
static int g_h = 0;

static HRESULT EnsureReadbackStaging()
{
    if (!g_ioTex || !g_device) return E_FAIL;
    if (g_stagingReadback) return S_OK;

    D3D11_TEXTURE2D_DESC src{};
    g_ioTex->GetDesc(&src);

    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = src.Width;
    desc.Height = src.Height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = src.Format;                 // must be R16_UINT
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;

    return g_device->CreateTexture2D(&desc, nullptr, g_stagingReadback.GetAddressOf());
}


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


        // CUDA device select
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

        int enableBlur = p.reserved[0];
        int enableInvert = p.reserved[1];
        int enableThreshold = p.reserved[2];
        int thresholdValue = p.reserved[3];

        cr = CudaProcessArray_R16_Inplace(
            ioArr, g_w, g_h,
            p.window, p.level,
            p.enableEdge,
            enableBlur,
            enableInvert,
            enableThreshold,
            thresholdValue);

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

        g_stagingReadback.Reset();
        g_ioTex.Reset();

        g_context.Reset();
        g_device.Reset();

        g_w = 0;
        g_h = 0;

        return IPC_OK;
    }

    
    int32_t __cdecl IPC_ReadbackRaw16(void* dst, int32_t dstBytes)
    {
        if (!g_initialized.load()) return IPC_ERR_NOT_INITIALIZED;
        if (!g_ioTex || !g_context) return IPC_ERR_NOT_INITIALIZED;
        if (!dst || dstBytes <= 0) return IPC_ERR_INVALIDARG;

        const int32_t need = (int32_t)(g_w * g_h * 2);
        if (dstBytes < need) return IPC_ERR_INVALIDARG;

        HRESULT hr = EnsureReadbackStaging();
        if (FAILED(hr) || !g_stagingReadback) return IPC_ERR_INTERNAL;

        // GPU copy -> staging
        g_context->CopyResource(g_stagingReadback.Get(), g_ioTex.Get());
        g_context->Flush();

        D3D11_MAPPED_SUBRESOURCE mapped{};
        hr = g_context->Map(g_stagingReadback.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr)) return IPC_ERR_INTERNAL;

        // copy row by row (RowPitch may be larger than width*2)
        uint8_t* out = reinterpret_cast<uint8_t*>(dst);
        const int rowBytes = g_w * 2;
        const uint8_t* src = reinterpret_cast<const uint8_t*>(mapped.pData);
        const int srcPitch = (int)mapped.RowPitch;

        for (int y = 0; y < g_h; y++)
        {
            memcpy(out + y * rowBytes, src + y * srcPitch, rowBytes);
        }

        g_context->Unmap(g_stagingReadback.Get(), 0);
        return IPC_OK;
    }


} // extern "C"
