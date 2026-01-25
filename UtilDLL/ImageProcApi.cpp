#include "ImageProcApi.h"

#include <d3d11.h>
#include <dxgi1_2.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <atomic>
#include <mutex>
#include <algorithm>

using Microsoft::WRL::ComPtr;



// ============================================================================
// Global-less helpers (buffer -> device -> immediate context)
// ============================================================================
static int32_t GetDeviceAndImmediateContextFromBuffer(
    void* ioBufferPtr,
    ComPtr<ID3D11Buffer>& outBuf,
    ComPtr<ID3D11Device>& outDev,
    ComPtr<ID3D11DeviceContext>& outCtx)
{
    if (!ioBufferPtr) return IPC_ERR_INVALID_ARG;

    ID3D11Buffer* bufRaw = reinterpret_cast<ID3D11Buffer*>(ioBufferPtr);
    if (!bufRaw) return IPC_ERR_INVALID_ARG;

    outBuf = bufRaw; // ComPtr takes AddRef

    ComPtr<ID3D11Device> dev;
    bufRaw->GetDevice(dev.GetAddressOf());
    if (!dev) return IPC_ERR_INTERNAL;

    ComPtr<ID3D11DeviceContext> ctx;
    dev->GetImmediateContext(ctx.GetAddressOf());
    if (!ctx) return IPC_ERR_INTERNAL;

    outDev = dev;
    outCtx = ctx;
    return IPC_OK;
}

static HRESULT CreateStagingBuffer(
    ID3D11Device* dev,
    int32_t requiredBytes,
    UINT cpuAccessFlags,
    ComPtr<ID3D11Buffer>& outStaging)
{
    if (!dev) return E_FAIL;
    if (requiredBytes <= 0) return E_INVALIDARG;

    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = (UINT)requiredBytes;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = cpuAccessFlags; // D3D11_CPU_ACCESS_WRITE or READ
    desc.MiscFlags = 0;
    desc.StructureByteStride = 0;

    outStaging.Reset();
    return dev->CreateBuffer(&desc, nullptr, outStaging.GetAddressOf());
}

// ============================================================================
// Global-less exported APIs
// ============================================================================
int32_t __cdecl IPC_UploadRaw16ToBufferEx(
    int32_t /*gpuId*/, void* ioBufferPtr,
    const void* src, int32_t srcBytes,
    int32_t width, int32_t height)
{
    if (!src) return IPC_ERR_INVALID_ARG;
    if (width <= 0 || height <= 0) return IPC_ERR_INVALID_STATE;

    // required bytes for R16 image
    const int32_t required = width * height * 2;
    if (srcBytes < required) return IPC_ERR_INVALID_ARG;

    // derive device/context from the buffer itself
    ComPtr<ID3D11Buffer> ioBuf;
    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext> ctx;
    int32_t rc = GetDeviceAndImmediateContextFromBuffer(ioBufferPtr, ioBuf, dev, ctx);
    if (rc != IPC_OK) return rc;

    // capacity check (safety)
    D3D11_BUFFER_DESC ioDesc{};
    ioBuf->GetDesc(&ioDesc);
    if ((int32_t)ioDesc.ByteWidth < required)
        return IPC_ERR_INVALID_STATE; // IO buffer too small

    // staging upload buffer (local, no globals)
    ComPtr<ID3D11Buffer> staging;
    HRESULT hr = CreateStagingBuffer(dev.Get(), required, D3D11_CPU_ACCESS_WRITE, staging);
    if (FAILED(hr) || !staging) return IPC_ERR_INTERNAL;

    D3D11_MAPPED_SUBRESOURCE mapped{};
    hr = ctx->Map(staging.Get(), 0, D3D11_MAP_WRITE, 0, &mapped);
    if (FAILED(hr) || !mapped.pData) return IPC_ERR_INTERNAL;

    // linear buffer copy
    memcpy(mapped.pData, src, required);

    ctx->Unmap(staging.Get(), 0);

    // staging -> IO
    ctx->CopyResource(ioBuf.Get(), staging.Get());
    return IPC_OK;
}

int32_t __cdecl IPC_ReadbackRaw16FromBufferEx(
    int32_t /*gpuId*/, void* ioBufferPtr,
    void* dst, int32_t dstBytes,
    int32_t width, int32_t height)
{
    if (!dst) return IPC_ERR_INVALID_ARG;
    if (width <= 0 || height <= 0) return IPC_ERR_INVALID_STATE;

    const int32_t required = width * height * 2;
    if (dstBytes < required) return IPC_ERR_INVALID_ARG;

    // derive device/context from the buffer itself
    ComPtr<ID3D11Buffer> ioBuf;
    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext> ctx;
    int32_t rc = GetDeviceAndImmediateContextFromBuffer(ioBufferPtr, ioBuf, dev, ctx);
    if (rc != IPC_OK) return rc;

    // capacity check (safety)
    D3D11_BUFFER_DESC ioDesc{};
    ioBuf->GetDesc(&ioDesc);
    if ((int32_t)ioDesc.ByteWidth < required)
        return IPC_ERR_INVALID_STATE;

    // staging readback buffer (local, no globals)
    ComPtr<ID3D11Buffer> staging;
    HRESULT hr = CreateStagingBuffer(dev.Get(), required, D3D11_CPU_ACCESS_READ, staging);
    if (FAILED(hr) || !staging) return IPC_ERR_INTERNAL;

    // IO -> staging
    ctx->CopyResource(staging.Get(), ioBuf.Get());
    ctx->Flush(); // ensure copy is visible before Map (conservative)

    D3D11_MAPPED_SUBRESOURCE mapped{};
    hr = ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr) || !mapped.pData) return IPC_ERR_INTERNAL;

    memcpy(dst, mapped.pData, required);

    ctx->Unmap(staging.Get(), 0);
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


    static int32_t CreateDeviceForGpu(int32_t gpuId, ID3D11Device** dev, ID3D11DeviceContext** ctx)
    {
        // 既にあなたのコードに「gpuIdに対応するIDXGIAdapterを選ぶ」関数があるならそれを使う
        ComPtr<IDXGIFactory1> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return -1;

        ComPtr<IDXGIAdapter1> adapter;
        if (FAILED(factory->EnumAdapters1((UINT)gpuId, &adapter))) return -2;

        UINT flags = 0;
#ifdef _DEBUG
        flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
        D3D_FEATURE_LEVEL fl;
        HRESULT hr = D3D11CreateDevice(
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
        if (FAILED(hr)) return -3;
        return 0;
    }



    IPC_API void __cdecl IPC_DestroyIoSharedHandle(void* sharedHandle)
    {
        sharedHandle = nullptr;
    }


    IPC_API void* __cdecl IPC_CreateIoBuffer(int32_t gpuId, int32_t width, int32_t height)
    {

        if (width <= 0 || height <= 0)
        {
            //SetErr(E_INVALIDARG, "Invalid width/height");
            return nullptr;
        }

        // ここは IPC_Init と同じデバイス生成ロジックを使うのが重要
        ComPtr<ID3D11Device> dev;
        ComPtr<ID3D11DeviceContext> ctx;
        HRESULT hr = CreateDeviceOnAdapterIndex((int)gpuId, dev.GetAddressOf(), ctx.GetAddressOf());
        if (FAILED(hr))
        {
            //SetErr(hr, "CreateDeviceOnAdapterIndex failed");
            return nullptr;
        }

        // 例：R16画像を想定（必要ならBgra32等に合わせて変更）
        const uint64_t bytes = (uint64_t)width * (uint64_t)height * 2ULL;
        if (bytes == 0 || bytes > (uint64_t)UINT32_MAX)
        {
            //SetErr(E_INVALIDARG, "Buffer size overflow");
            return nullptr;
        }

        D3D11_BUFFER_DESC bd{};
        bd.ByteWidth = (UINT)bytes;
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;

        // Raw UAV/SRVを作りやすくする（CUDA登録側でも扱いやすいことが多い）
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;

        // Raw viewの要件：ByteWidth が 4の倍数推奨
        // R16画像(2bytes)だと (w*h*2) が 4の倍数にならない可能性があるので、
        // 必要なら 4バイト境界に丸める（API要件次第で選択）
        // ここでは安全側で丸める例：
        if ((bd.ByteWidth & 3u) != 0)
        {
            bd.ByteWidth = (bd.ByteWidth + 3u) & ~3u;
        }

        ComPtr<ID3D11Buffer> buf;
        hr = dev->CreateBuffer(&bd, nullptr, buf.GetAddressOf());
        if (FAILED(hr))
        {
            //SetErr(hr, "CreateBuffer failed");
            return nullptr;
        }

        // ★DLL境界で返すので AddRef して “受け取り側の参照” を作る
        // ComPtrの所有とは別に、返したポインタを受け取り側が保持してよいようにする
        ID3D11Buffer* raw = buf.Get();
        raw->AddRef();

        // この関数内の ComPtr はスコープアウトで Release されるが、
        // AddRef 済みなので返した参照は有効に残る
        return (void*)raw;
    }

    IPC_API void __cdecl IPC_ReleaseD3D11Resource(void* d3d11Resource)
    {
        // 受け取った側が最後に呼ぶ。AddRefした分をReleaseして寿命を正しく閉じる。
        IUnknown* unk = (IUnknown*)d3d11Resource;
        if (unk) unk->Release();
    }


} // extern "C"
