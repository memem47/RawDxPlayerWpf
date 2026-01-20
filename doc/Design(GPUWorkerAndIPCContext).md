# README / 設計書

## GPU Worker + IPC Context 設計（Single-thread GPU Executor）
### 目的

本モジュールは、**D3D11 / CUDA interop を含む GPU 処理を「必ず同一スレッドで直列実行」**するための基盤を提供する。

- GPU API は「呼び出しスレッド制約」や「同時アクセス制約」が強い

- UI スレッド / 任意スレッドからの呼び出しを許容しつつ、GPU 側は 単一スレッドに閉じ込める

- 呼び出し側には同期 API（SubmitAndWait）を提供し、実行の成否は int32_t エラーコードで返す

### コンポーネント概要
#### GpuWorker（single-thread executor）

- ワーカースレッド 1 本を生成し、タスクキュー（FIFO）に入ったジョブを 直列実行する。

- SubmitAndWait(fn) は

1. 必要ならスレッド起動
2. packaged_task と future でジョブを enqueue
3. notify_one() で起床
4. future.get() で完了まで待機
を行う。

- condition_variable::wait(lock, pred) を使用し、スプリアスウェイクアップに対して安全に待機する。

#### IpcContext（GPU thread-owned state）

- D3D11 デバイス/コンテキスト、IO リソース（shared texture / buffer）、staging、CUDA interop 登録ハンドルを保持する。

- 所有者は GPU worker スレッドのみ。他スレッドから直接触れない（データ競合・不正な API 呼び出しを防ぐ）。

- Reset() は CUDA interop の解除（unregister）→ CUDA キャッシュ解放 → D3D リソース解放の順で行い、再初期化の前提状態に戻す。

### スレッド・所有権（重要）

- g_ctx（IpcContext）は GPU worker スレッドの専有物。
- D3D11/CUDA interop は「登録・map/unmap・解除」の順序や、対象デバイスの制約があるため、登録されたリソースは解除まで生存させる必要がある。
- GpuWorker::Stop() は「stop フラグ設定 → notify → join」で、キューが空になるまで実行してから停止する（安全なシャットダウン）。

### 典型的な呼び出しフロー

1. （呼び出し元スレッド）SubmitAndWait([&]{ ... GPU処理 ... })
2. （GPU worker）必要なら g_ctx 初期化（device/context 作成、IO/staging 作成、CUDA interop 登録）
3. （GPU worker）map → kernel / copy → unmap
4. （呼び出し元スレッド）戻り値（int32_t）で成功/失敗を判断

### エラー・例外ポリシー

- GPU スレッド上のジョブは 例外を投げない（C API / P/Invoke 境界を想定）。
- 失敗はエラーコード（int32_t）で返し、必要に応じてログ出力する。

### 制約・注意点（Pitfalls）

- SubmitAndWait() は同期 API なので、多用すると呼び出し元をブロックする（UI スレッドからの連打に注意）。
- Stop() は DLL unload / プロセス終了前に必ず呼ぶ（GPU リソース解放とスレッド終了順序が崩れるとクラッシュ要因）。
- D3D11 interop リソースは cudaGraphicsD3D11RegisterResource → map/unmap → cudaGraphicsUnregisterResource のライフサイクルを守る。

### 参考文献（一次情報中心）

- C++ std::condition_variable::wait（述語付き wait とスプリアスウェイクアップ）
- NVIDIA CUDA Runtime API: Direct3D 11 Interoperability（D3D11-CUDA interop）
- cudaGraphicsD3D11RegisterResource / unregister の説明（登録により参照カウントが増える等）

## cudaMallocAsync + stream 統一版への拡張コメント（差し込み用）

ここからは「CudaInterop 側を Context 化し、cudaMallocAsync/cudaFreeAsync と 単一 stream（例: cudaStream_t stream;）で統一する」前提で、**コードに追加すべき“製品レベルのコメント”**です。

### 2-1. 設計方針（コメントとして入れるべき要点）
#### A) “Stream-ordered allocator” の前提を明示する

cudaMallocAsync は stream 順序で allocate/free が成立するため、アクセス順序を破ると未定義動作（use-after-free 等）になる。

さらに、従来の cudaMalloc/cudaFree は全ストリーム同期を引き起こし得るが、stream-ordered allocator はこれを回避しやすい。

👉 なのでコメントにこう書く（要旨）：

- 「このポインタは この stream 上の work にのみ関連付けられている」
- 「free は cudaFreeAsync(ptr, stream) を使い、同一 stream の順序保証に依存する」
- 「他 stream / host からのアクセスを混ぜない」

#### B) “1 worker thread = 1 CUDA stream” を不変条件として宣言する

GPU worker はそもそも single-thread executor なので、
- GPU worker thread の中だけで stream を作る
- その stream は CudaContext に保持し、全カーネル/コピー/alloc/free をそこへ流す

👉 コメントにこう書く（要旨）：

- 「stream は GPU worker スレッドに束縛される（作成/破棄/使用は同スレッドのみ）」
- 「stream を跨ぐ同期は（基本）入れない。必要なら event に統一する」

### C) free の前に “アクセス完了” を呼び出し側が保証する必要がある点

CUDA Runtime API のメモリ管理では、cudaFree/cudaFreeAsync 呼び出し前に「当該メモリへのアクセスが完了していること」を呼び出し側が保証する必要がある、という注意がある（特に async allocator 関連）。

→ 設計としては「同一 stream に載せる」ことで順序保証を成立させるのが筋。

### 2-2. 具体的に差し込むコメント例（IpcContext / CudaInterop 側）
#### IpcContext 側に追加するなら（例）
```dcpp
// CUDA execution stream (created and used ONLY on the GPU worker thread)
//
// DESIGN INVARIANT:
//  - All CUDA work (kernels, async copies, interop map/unmap sequencing,
//    and stream-ordered allocations) must be issued to this single stream.
//  - This guarantees ordering without cross-stream synchronization.
//  - Do NOT access stream-ordered allocations from any other stream.
//
// RATIONALE:
//  - cudaMalloc/cudaFree may introduce device-wide synchronization,
//    while stream-ordered allocator (cudaMallocAsync/cudaFreeAsync)
//    enables allocation/free to be ordered with work in this stream.  (see refs)
cudaStream_t stream = nullptr;
```
#### CudaInterop の “Context 化”クラスの先頭コメント（例）
```cpp
/*
 * CudaContext (GPU-worker-thread owned)
 *
 * Owns:
 *  - a dedicated cudaStream_t for all CUDA work
 *  - (optional) a cudaMemPool_t configuration if customizing pool behavior
 *
 * Stream-Ordered Allocator Policy:
 *  - Use cudaMallocAsync/cudaFreeAsync on this stream ONLY.
 *  - Any access to async-allocated memory MUST occur between the
 *    stream-ordered allocation and free operations; otherwise undefined behavior.
 *
 * Threading:
 *  - Create/destroy/use only on the GPU worker thread.
 */
```
#### CudaReleaseCache() のコメントを “mallocAsync 前提”に更新
```cpp
// Release allocator/cache resources associated with stream-ordered allocations.
//
// NOTE:
//  - With cudaMallocAsync, the allocator uses a memory pool.
//  - Cache release should be done only after the stream has been drained
//    (i.e., no outstanding work that may touch pooled allocations).
//  - Do NOT call from non-GPU threads.
```
### 2-3. “統一 stream” を前提にした運用ルール（README に追記推奨）

- interop の map/unmap、カーネル、copy、alloc/free は すべて同一 stream に投入する
- CPU 側で結果が必要な境界（SubmitAndWait の戻り）では、必要に応じて stream 同期（例: cudaStreamSynchronize(stream) もしくは event wait）
- cudaFreeAsync は 同一 stream 上の順序保証に依存するため、別 stream で触る設計にしない（未定義動作リスク）