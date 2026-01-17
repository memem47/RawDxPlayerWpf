## ImageProcTestRunner

`ImageProcTestRunner` は、
C++ / CUDA 画像処理 DLL（ImageProcCudaDLL）を GUI なしで実行・検証するための CLI テストランナーです。
- CSV で指定した画像・処理パラメータを用いて処理を実行
- 処理結果を golden（基準画像）と比較して一致度を評価
- または golden を新規生成するモードで結果を書き出し
を行うことができます。

### 主な用途
- GPU / CUDA 画像処理カーネルの 単体テスト
- アルゴリズム変更時の 回帰テスト
- golden 画像の生成・更新
- GUI とは独立した 自動テスト / CI 実行

### ディレクトリ構成（例）
```csharp
repo-root/
├─ ImageProcCudaDLL/        # C++/CUDA DLL
├─ tools/
│   └─ ImageProcTestRunner/
│       ├─ ImageProcTestRunner.exe
│       ├─ README.md
│       ├─ csv.h
│       ├─ raw16.h
│       ├─ metrics.h
│       └─ main.cpp
└─ tests/
    ├─ test_cases.csv
    ├─ input/
    │   └─ lena_512x512_u16.raw
    ├─ golden/
    │   └─ lena_wlww.raw
    └─ out/
        ├─ results.csv
        └─ *.raw
```
### CSV テストケース形式
`tests/test_cases.csv` の 1 行が 1 テストケースに対応します。

#### カラム一覧
カラム名|内容
-|-
name|テスト名
input|入力 RAW16 画像
width,height|画像サイズ
window,level|WL / WW
enable_edge|Sobel (0/1)
enable_blur|Blur (0/1)
enable_invert|Invert (0/1)
enable_threshold|Threshold (0/1)
threshold_value|Threshold 値
golden|golden 画像のパス
metric|評価指標（exact / max_abs / mae / psnr）
pass_value|合格判定の閾値
note|任意メモ

### 実行方法
#### 基本形（比較モード）

CSV に定義されたテストをすべて実行し、
結果を golden と比較して PASS / FAIL を判定します。
```bat
ImageProcTestRunner.exe tests/test_cases.csv --gpu 0 --outdir tests/out
```
- tests/out/results.csv に結果が出力されます
- 各テストの出力画像は tests/out/<name>.raw に保存されます

#### golden 生成モード（比較なし）

指定したパラメータで処理を行い、
CSV の `golden` 列で指定されたパスに結果を直接書き込みます。
```bat
ImageProcTestRunner.exe tests/test_cases.csv --gpu 0 --generate-golden
```

このモードでは：
- golden 画像は 上書き生成
- 比較は行いません
- results.csv には GENERATED として記録されます

典型的な用途：
- 初回の golden 作成
- アルゴリズム変更後の golden 更新

#### 出力ディレクトリ指定
```bat
ImageProcTestRunner.exe tests/test_cases.csv --gpu 0 --outdir tests/out
```
#### 出力保存を無効化（比較のみ）
```bat
ImageProcTestRunner.exe tests/test_cases.csv --gpu 0 --no-save-out
```
#### VS2022 でのデバッグ実行（F5）
設定場所
プロジェクトのプロパティ
→ 構成プロパティ
→ デバッグ

#### コマンド引数（例）
```bat
tests/test_cases.csv --gpu 0 --outdir tests/out
```
#### 作業ディレクトリ（重要）
$(SolutionDir)

これにより、tests/... の相対パスが正しく解決されます。

#### 評価指標（metric）
metric|判定内容
-|-
exact|完全一致（memcmp）
max_abs|最大絶対誤差 ≤ pass_value
mae|Mean Absolute Error ≤ pass_value
psnr|PSNR ≥ pass_value (dB)

#### デバッグについて

- `ImageProcCudaDLL.dll` と 対応する `.pdb` を
`ImageProcTestRunner.exe` と同じフォルダに配置してください

- Debug / x64 構成でビルドすることで、
    - DLL 内部
    - CUDA 呼び出し直前
      まで ステップイン可能です

#### 典型的な運用フロー

1. golden 生成

```bat
ImageProcTestRunner.exe tests/test_cases.csv --gpu 0 --generate-golden
```

2. golden を Git にコミット
3. 通常テスト実行

```bat
ImageProcTestRunner.exe tests/test_cases.csv --gpu 0 --outdir tests/out
```

4. FAIL が出たら tests/out/*.raw を確認

### 備考
- 本ツールは GUI に依存せず、DLL の IPC API のみを使用します
- CSV 駆動のため、テストケースの追加・変更が容易です
- CI（GitHub Actions 等）での自動実行にも適しています