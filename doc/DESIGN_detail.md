# RawDxPlayerWpf + ImageProcCudaDLL 詳細設計書（ドラフト）
作成日: 2026-01-12

> 注: DLL側はアップロードされたソースに基づき、実装レベルまで踏み込みます。  
> WPF側は「I/Fとフレーム処理の流れ」を中心に記述します（WPF全ファイルが一部期限切れのため）。

---

## 1. ImageProcCudaDLL 詳細設計

### 1.1 ファイル構成と責務
- `ImageProcApi.h`
  - DLL公開API、`IPC_Params` 定義
  - CUDA側エントリ関数宣言（interop/processing）
- `ImageProcApi.cpp`
  - D3D11デバイス作成・共有テクスチャOpen
  - CUDA登録（`cudaGraphicsD3D11RegisterResource`）
  - `IPC_SetParams` の軽量コピー
  - `IPC_Execute` の map→kernel→unmap 実行
- `CudaInterop.cu`
  - CUDA interop（register/map/unmap）
  - kernel群（WL/WW、Sobel）
  - R16 input -> BGRA output の処理関数
- `pch.h/.cpp`, `framework.h`, `dllmain.cpp`
  - 既定のDLL構成

---

### 1.2 主要データ（グローバル）
- `cudaGraphicsResource* g_cudaIn / g_cudaOut`
  - D3D11共有テクスチャのCUDA登録ハンドル
- `ComPtr<ID3D11Texture2D> g_inputTex / g_outputTex`
  - 共有テクスチャ実体
- `IPC_Params g_params`
  - 最新パラメータ（GUI更新により頻繁に更新）
- `g_w, g_h`
  - テクスチャサイズ

**スレッド安全性**
- `IPC_SetParams`: 単純コピー（アトミックではないが、構造体サイズが小さく実運用で問題が出にくい）
- `IPC_Execute`: 実行開始時にローカルへ `IPC_Params p = g_params;` とコピーして使用  
  → 実行中にGUIが更新しても影響しない

---

### 1.3 初期化シーケンス（IPC_Init）
1. `IPC_Shutdown()` で既存リソースを破棄
2. 指定 `gpuId` の DXGI adapter から D3D11 device/context を作成
3. `OpenSharedResource` で入力/出力の `ID3D11Texture2D` を取得
4. format検証
   - 入力: `DXGI_FORMAT_R16_UINT`
   - 出力: `DXGI_FORMAT_B8G8R8A8_UNORM`
5. `cudaSetDevice(gpuId)`
6. `cudaGraphicsD3D11RegisterResource` を input/output に対して実行

**エラー方針**
- 主要失敗点ごとに -2, -3, -5, -6, -1100... のように区別して返す  
  → WPF側でユーザー向けメッセージ化推奨

---

### 1.4 実行シーケンス（IPC_Execute）
1. `CudaMapGetArraysMapped(g_cudaIn, g_cudaOut, &inArr, &outArr)`
   - `cudaGraphicsMapResources`（入力・出力をmap）
   - `cudaGraphicsSubResourceGetMappedArray` で `cudaArray` を取得
   - **重要: map状態のまま返す**
2. ローカルへ `p = g_params` をコピー
3. `CudaProcessArrays_R16_To_BGRA(inArr, outArr, w, h, p.window, p.level, p.enableEdge)`
4. `CudaUnmapResources` で unmap（出力→入力の順）

---

### 1.5 CUDA処理設計（CudaInterop.cu）

#### 1.5.1 リソース
- 入力: `cudaArray` を `cudaTextureObject_t` として参照（Point sampling, element type）
- 出力: `cudaArray` を `cudaSurfaceObject_t` として参照（`surf2Dwrite`）

#### 1.5.2 kernel
- `WLWWKernel`
  - R16値 → WL/WWで0..255へマップ → BGRA書き込み
- `Sobel16Kernel`
  - 近傍9点をWL/WW正規化（8bit化）してからSobel
  - `abs(gx)+abs(gy)` を 0..255 へ clamp

#### 1.5.3 ブロック/グリッド
- block: 16x16
- grid: `(w+15)/16, (h+15)/16`

---

### 1.6 ビルド設計（ImageProcCudaDLL.vcxproj）
- CUDA 12.2 Build Customization を使用
- `.cu` は `CudaInterop.cu` を `CudaCompile` としてビルド
- include/lib は `$(CUDA_PATH_V12_2)` を参照（環境依存）

---

## 2. WPF側（概要レベルの詳細）

### 2.1 主要責務
- Input shared texture (R16) へ RAW16 を upload
- Output shared texture (BGRA) を表示
- GUIパラメータ変更時に `IPC_SetParams`
- 再生ループで毎フレーム `IPC_Execute`

### 2.2 フレーム処理（推奨の分岐）
- DLL ON:
  - upload RAW16 → setParams（必要時）→ execute → 表示
- DLL OFF:
  - CPU WL/WW → outputへupload → 表示

---

## 3. 追加推奨（品質/保守）
- `IPC_Params` を version/size で厳密に検証（既に実装済み）
- 例外時も unmap が確実に走るよう RAII もしくは try/finally 相当で保護
- WPF側は Slider連打対策で `IPC_SetParams` を 30ms程度で間引き可能

