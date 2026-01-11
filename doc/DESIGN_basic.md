# RawDxPlayerWpf + ImageProcCudaDLL 基本設計書（ドラフト）
作成日: 2026-01-12

> 注: この設計書は、今回アップロードされた **ImageProcCudaDLL（C++/CUDA DLL）** のプロジェクト一式と、これまでの要件説明に基づいて作成しています。  
> WPF側（RawDxPlayerWpf）の全ファイルがこちらの環境で一部期限切れになっているため、WPF側は「構成とI/F中心」の記述です。必要なら WPF側一式の再アップでより厳密に更新します。

---

## 1. 目的
- RAW16（.raw/.bin）連番をWPFで動画再生し、Direct3D11の共有テクスチャを介して **CUDA画像処理**を適用した結果を表示する。
- GUIから **WL/WW（Window/Level）** と **Edge（Sobel）** をリアルタイムに変更し、`IPC_SetParams` でDLLへ反映する。

---

## 2. システム構成

### 2.1 アプリ（WPF）
- 名称: RawDxPlayerWpf
- 技術: WPF / .NET Framework 4.8 / SharpDX（D3D11）
- 主な責務:
  - RAW16ファイルの選択、同フォルダ連番の列挙
  - 再生制御（タイマーでフレーム更新）
  - D3D11共有テクスチャ生成（Input/Output）
  - DLL呼び出し（P/Invoke）
  - GUI（左:表示、右:パラメータ）

### 2.2 DLL（C++/CUDA）
- 名称: ImageProcCudaDLL
- 技術: C++ / CUDA 12.2 / CUDA Graphics Interop（D3D11）
- 主な責務:
  - 共有テクスチャをCUDAに登録し、map/unmapで `cudaArray` を取得
  - `IPC_Execute()` で R16入力から BGRA出力を生成（WL/WW or Sobel）
  - `IPC_SetParams()` でパラメータを保存し、実行時に参照

---

## 3. 入出力仕様（D3D11共有テクスチャ）

### 3.1 入力（WPF → DLL）
- フォーマット: **DXGI_FORMAT_R16_UINT**
- 内容: RAW16（little-endian）をそのままGPUにアップロードしたテクスチャ

### 3.2 出力（DLL → WPF）
- フォーマット: **DXGI_FORMAT_B8G8R8A8_UNORM**
- 内容: 表示用BGRA8（WL/WW適用後、またはEdge）

---

## 4. DLL API（外部I/F）

### 4.1 公開API一覧
- `IPC_Init()`
- `IPC_SetParams()`
- `IPC_Execute()`
- `IPC_Shutdown()`

### 4.2 APIの責務
- `IPC_Init(gpuId, inSharedHandle, outSharedHandle)`
  - D3D11デバイス生成（指定GPU）
  - 共有テクスチャをOpen
  - 入力=R16, 出力=BGRAの整合性を検証
  - CUDAデバイス選択
  - CUDAにD3D11リソース登録
- `IPC_SetParams(params)`
  - **軽量コピーのみ**（GUIスライダー連打に耐える）
- `IPC_Execute()`
  - map → `cudaArray` 取得 → kernel実行 → unmap
  - `enableEdge` により WL/WW と Sobel を切替
- `IPC_Shutdown()`
  - CUDA登録解除、D3Dリソース解放

---

## 5. 主要ユースケース

### 5.1 再生（DLL ON）
1. WPFがRAW16を読み込み、Input(R16)へupload
2. GUIパラメータ変更時に `IPC_SetParams`
3. 毎フレーム `IPC_Execute`
4. WPFがOutput(BGRA)を表示

### 5.2 再生（DLL OFF）
- WPF側でCPU WL/WW → BGRA を生成し、Outputへuploadして表示（保険パス）

---

## 6. プロジェクトビルド概要（ImageProcCudaDLL）
- 構成: Debug/Win32, Release/Win32, Debug/x64, Release/x64
- CUDA Build Customization:
  - props: CUDA 12.2.props
  - targets: CUDA 12.2.targets
- CUDAソース:
  - CudaInterop.cu

---

## 7. 今後の拡張方針（高レベル）
- 画像処理を増やしやすい「処理チェーン」化（WL/WW → Edge → …）
- パラメータ構造体のversion管理（後方互換）
- 表示を readback から D3DImage 直描画へ（CPUコピー削減）
