#define IMAGEPROCCUDADLL_EXPORTS
#include "ImageProcApi.h"

#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <atomic>
#include <mutex>

using Microsoft::WRL::ComPtr;

static cudaGraphicsResource* g_cudaIn = nullptr;
static cudaGraphicsResource* g_cudaOut = nullptr;

static ComPtr<ID3D11Device>        g_device;
static ComPtr<ID3D11DeviceContext> g_context;
static ComPtr<ID3D11Texture2D>     g_inputTex;
static ComPtr<ID3D11Texture2D>     g_outputTex;

static ComPtr<ID3D11Texture2D>     g_stagingReadback;


static std::atomic<bool> g_initialized{ false };
static IPC_Params g_params{};
static std::mutex g_mtx;

static int g_w = 0;
static int g_h = 0;

static HRESULT EnsureReadbackStaging()
{
    if (!g_inputTex || !g_device) return E_FAIL;
    if (g_stagingReadback) return S_OK;

    D3D11_TEXTURE2D_DESC src{};
    g_inputTex->GetDesc(&src);

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

        // reserved[0] = enablePostFilter (default ON so you can see it without changing C# immediately)
        p.reserved[0] = 1;

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
        if (!g_cudaIn || !g_cudaOut) return IPC_ERR_NOT_INITIALIZED;

        void* inArr = nullptr;
        void* outArr = nullptr;

        // Map and get arrays (keep mapped until after kernel)
        int cr = CudaMapGetArraysMapped(g_cudaIn, g_cudaOut, &inArr, &outArr);
        if (cr != 0) return -1300 - cr;

        // Copy params locally (avoid race)
        IPC_Params p;
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            p = g_params;
        }

        int enablePostFilter = p.reserved[0]; // 0/1

        // Run kernel(s): in(R16) -> out(R16) with optional second stage
        cr = CudaProcessArrays_R16_To_R16(
            inArr, outArr, g_w, g_h,
            p.window, p.level,
            p.enableEdge,
            enablePostFilter);

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

        g_stagingReadback.Reset();
        g_inputTex.Reset();
        g_outputTex.Reset();
        g_context.Reset();
        g_device.Reset();

        g_w = 0;
        g_h = 0;

        return IPC_OK;
    }

    
    int32_t __cdecl IPC_ReadbackRaw16(void* dst, int32_t dstBytes)
    {
        if (!g_initialized.load()) return IPC_ERR_NOT_INITIALIZED;
        if (!g_inputTex || !g_context) return IPC_ERR_NOT_INITIALIZED;
        if (!dst || dstBytes <= 0) return IPC_ERR_INVALIDARG;

        const int32_t need = (int32_t)(g_w * g_h * 2);
        if (dstBytes < need) return IPC_ERR_INVALIDARG;

        HRESULT hr = EnsureReadbackStaging();
        if (FAILED(hr) || !g_stagingReadback) return IPC_ERR_INTERNAL;

        // GPU copy -> staging
        g_context->CopyResource(g_stagingReadback.Get(), g_inputTex.Get());
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
