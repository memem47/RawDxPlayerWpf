#pragma once
#include <stdint.h>

#ifdef IMAGEPROCCUDADLL_EXPORTS
#define IPC_API __declspec(dllexport)
#else
#define IPC_API __declspec(dllimport)
#endif

extern "C" {

    // 失敗時の原因切り分けがしやすいように、HRESULT相当ではなく int を返す（0=OK）
    enum IPC_Result : int32_t {
        IPC_OK = 0,
        IPC_ERR_INVALIDARG = 1,
        IPC_ERR_NOT_INITIALIZED = 2,
        IPC_ERR_INTERNAL = 3
    };

    // パラメータ構造体（後で拡張しても ABI を壊しにくいように version/size を先頭に置く）
#pragma pack(push, 1)
    struct IPC_Params
    {
        uint32_t sizeBytes;   // sizeof(IPC_Params)
        uint32_t version;     // 1

        // 例：階調変換の window/level
        int32_t window;
        int32_t level;

        // 例：エッジ抽出ON/OFFなど（後で増やす）
        int32_t enableEdge;

        int32_t reserved[8];  // 将来拡張用
    };
#pragma pack(pop)

    // A: 初期化
    IPC_API int32_t IPC_Init(int32_t gpuId, void* inSharedHandle, void* outSharedHandle);

    // B: パラメータ設定
    IPC_API int32_t IPC_SetParams(const IPC_Params* p);

    // C: 実行
    IPC_API int32_t IPC_Execute();

    // D: 終了
    IPC_API int32_t IPC_Shutdown();

}
