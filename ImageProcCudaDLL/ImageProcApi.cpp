#include "ImageProcApi.h"

#include <d3d11.h>
#include <dxgi1_2.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <atomic>
#include <mutex>

#include <thread>
#include <condition_variable>
#include <queue>
#include <future>
#include <functional>

using Microsoft::WRL::ComPtr;

// ------------------------------
// GPU Worker (single-thread executor)
// ------------------------------
class GpuWorker
{
public:
    void Start()
    {
        std::lock_guard<std::mutex> lk(m_);
        if (running_) return;
        stop_ = false;
        th_ = std::thread([this] { Run(); });
        running_ = true;
    }

    void Stop()
    {
        {
            std::lock_guard<std::mutex> lk(m_);
            if (!running_) return;
            stop_ = true;
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
        running_ = false;
    }

    int32_t SubmitAndWait(std::function<int32_t()> fn)
    {
        Start();
        std::packaged_task<int32_t()> task(std::move(fn));
        auto fut = task.get_future();
        {
            std::lock_guard<std::mutex> lk(m_);
            q_.push(std::move(task));
        }
        cv_.notify_one();
        return fut.get();
    }

private:
    void Run()
    {
        while (true)
        {
            std::packaged_task<int32_t()> task;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
                if (stop_ && q_.empty()) break;
                task = std::move(q_.front());
                q_.pop();
            }
            task(); // execute job
        }
    }

    std::thread th_;
    std::mutex m_;
    std::condition_variable cv_;
    std::queue<std::packaged_task<int32_t()>> q_;
    bool stop_ = false;
    bool running_ = false;
};

static GpuWorker g_worker;


static cudaGraphicsResource* g_cudaIO = nullptr;

static ComPtr<ID3D11Device>        g_device;
static ComPtr<ID3D11DeviceContext> g_context;

static ComPtr<ID3D11Texture2D>     g_ioTex;
static ComPtr<ID3D11Buffer>   g_ioBuf;

static ComPtr<ID3D11Texture2D>     g_stagingUpload;
static ComPtr<ID3D11Texture2D>     g_stagingReadback;

static ComPtr<ID3D11Buffer> g_stagingUploadBuf;
static ComPtr<ID3D11Buffer> g_stagingReadbackBuf;

static ComPtr<ID3D11Texture2D>     g_ownedSharedTex;  // Createした実体（Init前）

static std::atomic<bool> g_initialized{ false };
static IPC_Params g_params{};
static std::mutex g_mtx;

static int g_w = 0;
static int g_h = 0;

static HRESULT EnsureUploadStaging()
{
    if (g_stagingUpload) return S_OK;
    if (!g_device || !g_ioTex) return E_FAIL;

    D3D11_TEXTURE2D_DESC desc{};
    g_ioTex->GetDesc(&desc);

    // CPU書き込み用 staging
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.MiscFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    return g_device->CreateTexture2D(&desc, nullptr, g_stagingUpload.GetAddressOf());
}

static HRESULT EnsureUploadStagingBuffer(int32_t requiredBytes)
{
    if (!g_device) return E_FAIL;
    if (requiredBytes <= 0) return E_INVALIDARG;

    if (g_stagingUploadBuf)
    {
        D3D11_BUFFER_DESC bd{};
        g_stagingUploadBuf->GetDesc(&bd);
        if ((int32_t)bd.ByteWidth >= requiredBytes)
            return S_OK;

        g_stagingUploadBuf.Reset();
    }

    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = (UINT)requiredBytes;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    desc.MiscFlags = 0;
    desc.StructureByteStride = 0;

    return g_device->CreateBuffer(&desc, nullptr, g_stagingUploadBuf.GetAddressOf());
}


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

static HRESULT EnsureReadbackStagingBuffer(int32_t requiredBytes)
{
    if (!g_device) return E_FAIL;
    if (requiredBytes <= 0) return E_INVALIDARG;

    if (g_stagingReadbackBuf)
    {
        D3D11_BUFFER_DESC bd{};
        g_stagingReadbackBuf->GetDesc(&bd);
        if ((int32_t)bd.ByteWidth >= requiredBytes)
            return S_OK;

        g_stagingReadbackBuf.Reset();
    }

    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = (UINT)requiredBytes;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;
    desc.StructureByteStride = 0;

    return g_device->CreateBuffer(&desc, nullptr, g_stagingReadbackBuf.GetAddressOf());
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

static int32_t Shutdown_Impl()
{
    g_initialized.store(false);

    if (g_cudaIO) { CudaUnregister(g_cudaIO); g_cudaIO = nullptr; }

    CudaReleaseCache();

    g_stagingUpload.Reset();
    g_stagingReadback.Reset();
    g_stagingUploadBuf.Reset();
    g_stagingReadbackBuf.Reset();

    g_ioTex.Reset();
    g_ioBuf.Reset();

    g_context.Reset();
    g_device.Reset();

    g_w = 0;
    g_h = 0;

    return IPC_OK;
}

static int32_t InitWithSharedHandle_Impl(int32_t gpuId, void* ioSharedHandle)
{
    if (!ioSharedHandle) return IPC_ERR_INVALIDARG;

    // 旧リソースをワーカースレッド内で解放
    Shutdown_Impl();

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
    p.reserved[0] = 0;

    g_params = p;

    g_initialized.store(true);
    return IPC_OK;
}

static int32_t InitWithIoBuffer_Impl(int32_t gpuId, void* ioBufferPtr)
{
    if (!ioBufferPtr) return IPC_ERR_INVALIDARG;

    Shutdown_Impl();

    IUnknown* unk = (IUnknown*)ioBufferPtr;

    ComPtr<ID3D11Buffer> buf;
    HRESULT hr = unk->QueryInterface(__uuidof(ID3D11Buffer), (void**)buf.GetAddressOf());
    if (FAILED(hr)) return -201;

    ComPtr<ID3D11Device> dev;
    buf->GetDevice(dev.GetAddressOf());
    if (!dev) return -202;

    ComPtr<ID3D11DeviceContext> ctx;
    dev->GetImmediateContext(ctx.GetAddressOf());
    if (!ctx) return -203;

    g_device = dev;
    g_context = ctx;
    g_ioBuf = buf;
    g_ioTex.Reset();

    D3D11_BUFFER_DESC bd{};
    g_ioBuf->GetDesc(&bd);
    if ((bd.BindFlags & D3D11_BIND_UNORDERED_ACCESS) == 0)
        return -204;

    int cr = CudaSetDeviceSafe((int)gpuId);
    if (cr != 0) return -1000 - cr;

    cr = CudaRegisterD3D11Buffer(g_ioBuf.Get(), &g_cudaIO);
    if (cr != 0) return -1200 - cr;

    // default params（Texture版Initと同様）
    IPC_Params p{};
    p.sizeBytes = sizeof(IPC_Params);
    p.version = 1;
    p.window = 4000;
    p.level = 2000;
    p.enableEdge = 0;
    p.reserved[0] = 0;
    g_params = p;

    g_initialized.store(true);
    return IPC_OK;
}

static int32_t SetParams_Impl(const IPC_Params* p)
{
    int32_t v = ValidateParams(p);
    if (v != IPC_OK) return v;
    if (!g_initialized.load()) return IPC_ERR_NOT_INITIALIZED;

    g_params = *p;
    g_w = p->width;
    g_h = p->height;
    return IPC_OK;
}

static int32_t Execute_Impl()
{
    if (!g_initialized.load()) return IPC_ERR_NOT_INITIALIZED;
    if (!g_cudaIO) return IPC_ERR_NOT_INITIALIZED;

    IPC_Params p = g_params;

    int enableBlur = p.reserved[0];
    int enableInvert = p.reserved[1];
    int enableThreshold = p.reserved[2];
    int thresholdValue = p.reserved[3];

    int cr = 0;
    int cr2 = 0;

    if (g_ioTex)
    {
        void* ioArr = nullptr;
        cr = CudaMapGetArrayMapped(g_cudaIO, &ioArr);
        if (cr != 0) return -1300 - cr;

        cr = CudaProcessArray_R16_Inplace(
            ioArr, g_w, g_h,
            p.window, p.level,
            p.enableEdge,
            enableBlur,
            enableInvert,
            enableThreshold,
            thresholdValue);

        cr2 = CudaUnmapResource(g_cudaIO);

        if (cr != 0)  return -1400 - cr;
        if (cr2 != 0) return -1500 - cr2;
        return IPC_OK;
    }

    if (g_ioBuf)
    {
        void* devPtr = nullptr;
        size_t mappedBytes = 0;

        cr = CudaMapGetPointerMapped(g_cudaIO, &devPtr, &mappedBytes);
        if (cr != 0) return -1310 - cr;

        const size_t required = (size_t)g_w * (size_t)g_h * 2u;
        if (mappedBytes < required)
        {
            CudaUnmapResource(g_cudaIO);
            return IPC_ERR_INVALID_STATE;
        }

        cr = CudaProcessBuffer_R16_Inplace(
            devPtr, g_w, g_h,
            p.window, p.level,
            p.enableEdge,
            enableBlur,
            enableInvert,
            enableThreshold,
            thresholdValue);

        cr2 = CudaUnmapResource(g_cudaIO);

        if (cr != 0)  return -1410 - cr;
        if (cr2 != 0) return -1510 - cr2;
        return IPC_OK;
    }

    return IPC_ERR_NOT_INITIALIZED;
}



extern "C" {

    int32_t __cdecl IPC_Init(int32_t gpuId, void* ioSharedHandle)
    {
        // 外スレッドで呼ばれてOK：中身はGPUスレッドで実行
        return g_worker.SubmitAndWait([=]() -> int32_t {
            return InitWithSharedHandle_Impl(gpuId, ioSharedHandle);
            });
    }

    int32_t __cdecl IPC_InitWithIoBuffer(int32_t gpuId, void* ioBufferPtr)
    {
        return g_worker.SubmitAndWait([=]() -> int32_t {
            return InitWithIoBuffer_Impl(gpuId, ioBufferPtr);
            });
    }


    int32_t __cdecl IPC_SetParams(const IPC_Params* p)
    {
        // スナップショット（呼び出し側が返った後にpが無効になる可能性があるのでコピー必須）
        if (!p) return IPC_ERR_INVALIDARG;
        IPC_Params copy = *p;

        return g_worker.SubmitAndWait([=]() -> int32_t {
            return SetParams_Impl(&copy);
            });
    }

    //int32_t __cdecl IPC_Execute()
    //{
    //    if (!g_initialized.load()) return IPC_ERR_NOT_INITIALIZED;
    //    if (!g_cudaIO) return IPC_ERR_NOT_INITIALIZED;

    //    void* ioArr = nullptr;
    //    int cr = CudaMapGetArrayMapped(g_cudaIO, &ioArr);
    //    if (cr != 0) return -1300 - cr;

    //    IPC_Params p;
    //    {
    //        std::lock_guard<std::mutex> lk(g_mtx);
    //        p = g_params;
    //    }

    //    int enableBlur = p.reserved[0];
    //    int enableInvert = p.reserved[1];
    //    int enableThreshold = p.reserved[2];
    //    int thresholdValue = p.reserved[3];

    //    cr = CudaProcessArray_R16_Inplace(
    //        ioArr, g_w, g_h,
    //        p.window, p.level,
    //        p.enableEdge,
    //        enableBlur,
    //        enableInvert,
    //        enableThreshold,
    //        thresholdValue);

    //    int cr2 = CudaUnmapResource(g_cudaIO);

    //    if (cr != 0) return -1400 - cr;
    //    if (cr2 != 0) return -1500 - cr2;
    //    return IPC_OK;
    //}

    int32_t __cdecl IPC_Execute()
    {
        return g_worker.SubmitAndWait([=]() -> int32_t {
            return Execute_Impl();
            });
    }


    int32_t __cdecl IPC_Shutdown()
    {
        // 1) release GPU/D3D/CUDA resources on worker
        int32_t r = g_worker.SubmitAndWait([=]() -> int32_t {
            return Shutdown_Impl();
            });

        // 2) stop worker thread (optional: keep alive if you prefer)
        g_worker.Stop();
        return r;
    }

    int32_t __cdecl IPC_UploadRaw16(const void* src, int32_t srcBytes)
    {
        if (!src) return IPC_ERR_INVALID_ARG;
        if (!g_device || !g_context || !g_ioTex) return IPC_ERR_NOT_INITIALIZED;
        if (g_w <= 0 || g_h <= 0) return IPC_ERR_INVALID_STATE;

        const int32_t required = g_w * g_h * 2;
        if (srcBytes < required) return IPC_ERR_INVALID_ARG;

        HRESULT hr = EnsureUploadStaging();
        if (FAILED(hr) || !g_stagingUpload) return IPC_ERR_INTERNAL;

        D3D11_MAPPED_SUBRESOURCE mapped{};
        hr = g_context->Map(g_stagingUpload.Get(), 0, D3D11_MAP_WRITE, 0, &mapped);
        if (FAILED(hr)) return IPC_ERR_INTERNAL;

        const uint8_t* in = reinterpret_cast<const uint8_t*>(src);
        const int rowBytes = g_w * 2;
        uint8_t* dst = reinterpret_cast<uint8_t*>(mapped.pData);
        const int dstPitch = (int)mapped.RowPitch;

        for (int y = 0; y < g_h; y++)
        {
            memcpy(dst + y * dstPitch, in + y * rowBytes, rowBytes);
        }

        g_context->Unmap(g_stagingUpload.Get(), 0);

        // staging -> IO
        g_context->CopyResource(g_ioTex.Get(), g_stagingUpload.Get());

        return IPC_OK;
    }

    int32_t __cdecl IPC_UploadRaw16ToBuffer(const void* src, int32_t srcBytes, int32_t width, int32_t height)
    {
        g_w = width;
        g_h = height;
        if (!src) return IPC_ERR_INVALID_ARG;
        if (!g_device || !g_context || !g_ioBuf) return IPC_ERR_NOT_INITIALIZED;
        if (g_w <= 0 || g_h <= 0) return IPC_ERR_INVALID_STATE;

        const int32_t required = g_w * g_h * 2;
        if (srcBytes < required) return IPC_ERR_INVALID_ARG;

        // IOバッファ容量チェック（安全）
        D3D11_BUFFER_DESC ioDesc{};
        g_ioBuf->GetDesc(&ioDesc);
        if ((int32_t)ioDesc.ByteWidth < required)
            return IPC_ERR_INVALID_STATE; // IOバッファが小さすぎる

        HRESULT hr = EnsureUploadStagingBuffer(required);
        if (FAILED(hr) || !g_stagingUploadBuf) return IPC_ERR_INTERNAL;

        D3D11_MAPPED_SUBRESOURCE mapped{};
        hr = g_context->Map(g_stagingUploadBuf.Get(), 0, D3D11_MAP_WRITE, 0, &mapped);
        if (FAILED(hr) || !mapped.pData) return IPC_ERR_INTERNAL;

        // Bufferは線形なので一発コピーでOK（RowPitchは使わない）
        memcpy(mapped.pData, src, required);

        g_context->Unmap(g_stagingUploadBuf.Get(), 0);

        // staging -> IO（Buffer）
        g_context->CopyResource(g_ioBuf.Get(), g_stagingUploadBuf.Get());

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

    int32_t __cdecl IPC_ReadbackRaw16FromBuffer(void* dst, int32_t dstBytes)
    {
        if (!dst) return IPC_ERR_INVALID_ARG;
        if (!g_device || !g_context || !g_ioBuf) return IPC_ERR_NOT_INITIALIZED;
        if (g_w <= 0 || g_h <= 0) return IPC_ERR_INVALID_STATE;

        const int32_t required = g_w * g_h * 2;
        if (dstBytes < required) return IPC_ERR_INVALID_ARG;

        // IOバッファ容量チェック（安全）
        D3D11_BUFFER_DESC ioDesc{};
        g_ioBuf->GetDesc(&ioDesc);
        if ((int32_t)ioDesc.ByteWidth < required)
            return IPC_ERR_INVALID_STATE;

        HRESULT hr = EnsureReadbackStagingBuffer(required);
        if (FAILED(hr) || !g_stagingReadbackBuf) return IPC_ERR_INTERNAL;

        // IO -> staging
        g_context->CopyResource(g_stagingReadbackBuf.Get(), g_ioBuf.Get());

        D3D11_MAPPED_SUBRESOURCE mapped{};
        hr = g_context->Map(g_stagingReadbackBuf.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr) || !mapped.pData) return IPC_ERR_INTERNAL;

        // Bufferは線形なので一発コピー
        memcpy(dst, mapped.pData, required);

        g_context->Unmap(g_stagingReadbackBuf.Get(), 0);

        return IPC_OK;
    }



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

    static HRESULT g_lastHr = S_OK;
    static const char* g_lastErr = "OK";
    static void SetErr(HRESULT hr, const char* msg) { g_lastHr = hr; g_lastErr = msg; }
    IPC_API void* __cdecl IPC_CreateIoSharedHandle(int32_t gpuId, int32_t width, int32_t height)
    {
        // 既存破棄（Destroyと同様、所有しているものだけ）
        g_ownedSharedTex.Reset();

        // ★ ここは IPC_Init と同じ関数を使う（すでに上にある）
        ComPtr<ID3D11Device> dev;
        ComPtr<ID3D11DeviceContext> ctx;
        HRESULT hr = CreateDeviceOnAdapterIndex((int)gpuId, dev.GetAddressOf(), ctx.GetAddressOf());
        if (FAILED(hr)) return nullptr;

        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = (UINT)width;
        desc.Height = (UINT)height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R16_UINT;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;

        // CUDA側で使うので最低限入れる
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;

        // ★ OpenSharedResource 互換のレガシー shared handle
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

        hr = dev->CreateTexture2D(&desc, nullptr, g_ownedSharedTex.GetAddressOf());
        if (FAILED(hr)) return nullptr;

        ComPtr<IDXGIResource> res;
        hr = g_ownedSharedTex.As(&res);
        if (FAILED(hr)) return nullptr;

        HANDLE h = nullptr;
        hr = res->GetSharedHandle(&h);
        if (FAILED(hr) || !h) return nullptr;

        return (void*)h;
    }


    IPC_API void __cdecl IPC_DestroyIoSharedHandle(void* sharedHandle)
    {
        sharedHandle = nullptr;
    }


    IPC_API void* __cdecl IPC_CreateIoBuffer(int32_t gpuId, int32_t width, int32_t height)
    {
        g_lastHr = S_OK;
        g_lastErr = "OK";

        if (width <= 0 || height <= 0)
        {
            SetErr(E_INVALIDARG, "Invalid width/height");
            return nullptr;
        }

        // ここは IPC_Init と同じデバイス生成ロジックを使うのが重要
        ComPtr<ID3D11Device> dev;
        ComPtr<ID3D11DeviceContext> ctx;
        HRESULT hr = CreateDeviceOnAdapterIndex((int)gpuId, dev.GetAddressOf(), ctx.GetAddressOf());
        if (FAILED(hr))
        {
            SetErr(hr, "CreateDeviceOnAdapterIndex failed");
            return nullptr;
        }

        // 例：R16画像を想定（必要ならBgra32等に合わせて変更）
        const uint64_t bytes = (uint64_t)width * (uint64_t)height * 2ULL;
        if (bytes == 0 || bytes > (uint64_t)UINT32_MAX)
        {
            SetErr(E_INVALIDARG, "Buffer size overflow");
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
            SetErr(hr, "CreateBuffer failed");
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

    IPC_API int32_t __cdecl IPC_GetLastHr()
    {
        return (int32_t)g_lastHr;
    }

    IPC_API const char* __cdecl IPC_GetLastErr()
    {
        return g_lastErr;
    }
} // extern "C"
