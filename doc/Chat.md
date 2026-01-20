## 設計の主目的
- 差し替え可能性：同じAPIで別実装に置き換えても動く（互換性の定義を明確化）
- リアルタイム性：遅延・スループット・メモリ・同期の約束を明確化
- 品質保証：テスト可能性、エラーの扱い、スレッド安全性、リソース管理のルール化
## 「何をすればいいか」をそのまま作業手順に
### 1) まず“設計”の成果物（納品物）を知る
設計では以下のセットを作る（全部を最初から完璧に作る必要はなく、最小→拡張でOK）。
A. 外部設計（API中心：差し替え前提で最重要）
API仕様書（契約書）
関数一覧、引数、戻り値、呼び出し順序、スレッド制約
エラーコード体系（何が起きたら何を返すか）
パフォーマンス要件（例：1080p/30fpsで平均Xms以下、最悪Yms以下）
利用シナリオ（正常系/異常系）
例：Initialize→SetParameters→Execute→Shutdown
例外：未初期化でExecuteしたら？サイズ違い入力は？GPU未対応環境は？
互換性ルール
バージョニング方針（ABI互換、構造体追加時のルールなど）
「置換可能」と見なす条件（出力差の許容範囲、丸め、境界処理）
B. 内部設計（実装を壊れにくくする）
アーキテクチャ図（責務分割）
C#↔P/Invoke境界、スレッド境界、GPU境界、メモリ境界を明確に
処理パイプライン設計
1フレーム処理の流れ、キュー、バッファ、同期（D3D/CUDA interop等）
スレッド設計
呼び出しスレッド要件、内部ワーカースレッド、排他、ロック方針
メモリ/リソース設計
誰が確保して誰が解放するか（所有権）
再利用バッファ、アロケーション頻度、リーク検出方針
ログ/診断設計
ログレベル、計測（フレーム時間、キュー深さ）、エラー原因の可観測性
C. 品質設計（ここが「品質を高めろ」の中身）
テスト設計
単体/結合/性能/長時間（リーク・累積遅延）テスト
“golden”比較、許容誤差、再現性（乱数/ノイズのシード）
受け入れ基準（Definition of Done）
「この条件を満たしたらOK」を事前に決める（後で揉めない）
1) いきなり全部やらない：最小の“設計パック”
あなたが「設計未経験」でも回る、最小構成はこれです（まずはこれだけ作ると“設計した”と言える）。
最小パック（まず1〜2日で作る）
API契約書（1〜3ページ）
呼び出し順序の状態遷移図（1枚）
1フレームのデータフロー図（1枚）
スレッド/所有権ルール（箇条書き）
性能目標＆計測方法（箇条書き）
テスト方針（何をどう確認するかの表）
これができると、以降の実装・レビュー・テストが全部この“契約”に紐づきます。
1) あなたの案件に合わせた「具体的に書くべき内容」
C#から呼ばれるリアルタイムDLLで特に重要な論点を、そのままドキュメントの見出しにします。
3.1 API（P/Invoke境界）で決めること
データ形式：入力/出力が byte* / ushort* / ID3D11Texture2D など何か、stride、エンディアン
呼び出しスレッド：
「全APIは同一スレッドから呼ぶ必要がある」or「スレッドセーフ」or「特定APIのみ排他」
リアルタイム制約：
Executeはブロッキングか？非同期か？（future/queue方式か）
エラー設計：
返り値（int error code）＋ GetLastErrorMessage() のような補助API
寿命：
Initializeで生成したコンテキストハンドルを全APIに渡す設計にするか（推奨：Handle方式）
3.2 置き換え前提（互換性）で決めること
出力の同等性：bit-exact要求？許容誤差？（特にGPUは差が出やすい）
パラメータの解釈：境界値・デフォルト値・単位（例：WL/WW、threshold）
例外ケースの振る舞い：入力不足、サイズ不一致、null、未初期化
3.3 リアルタイム（性能）で決めること
1フレームあたりの予算（例：33.3msのうちDLLは5ms以下）
メモリ確保禁止ルール（例：Execute内でnew/malloc禁止、初期化時にプール確保）
計測の仕方：DLL内で BeginFrame/EndFrame 計測してログ/統計を出す、など
1) “設計”の進め方（あなたのやり方を否定しない手順）
あなたは「まず動くものを作ってからドキュメント」タイプとのことなので、現実的にはこう進めるのが強いです。
API仕様を読み、未確定点を洗い出す（曖昧なまま実装すると地獄になる）
状態遷移図を書く（呼び出し順の契約を固定）
データフロー図を書く（1フレームで何がどこを流れるか）
スレッド/所有権ルールを書く（事故の9割がここ）
性能目標と“測り方”を書く（達成できたか判断できるように）
テスト観点を列挙する（golden、長時間、異常系）
ここまで書いたら実装（ただし設計は更新し続ける：設計＝生きた契約）
ポイントは、コードより先に「揉めるところ」だけ確定することです。全部を予見する必要はありません。
1) そのまま使える「設計書テンプレ（見出しだけ）」
GitHubに置く前提で、まずは docs/design.md みたいな1ファイルでOKです。
目的 / 対象範囲
用語
前提（対応OS、GPU要件、依存DLL、スレッドモデル）
外部インターフェース（API一覧）
状態遷移（Initialize/Execute/Shutdown）
データ仕様（入力/出力フォーマット、stride、アライン）
エラー仕様（エラーコード表、復帰可能性）
性能要件（目標、測定方法、ログ）
スレッド安全性（呼び出し制約、内部ワーカ）
メモリ/リソース（所有権、確保タイミング、上限）
内部アーキテクチャ（モジュール責務、依存関係）
テスト方針（単体/結合/性能/長時間、golden）
変更方針（互換性、バージョニング）
必要なら、あなたのDLLの現在API（Initialize/SetParameters/Execute/Shutdown、入出力形式、D3D/CUDAの使い方）を前提にして、上のテンプレを**あなたのプロジェクト用に具体語で埋めた“設計書の叩き台”**をこちらで一式作れます。今あるAPIシグネチャ（C#側のP/Invoke宣言かC++ヘッダ）を貼ってくれれば、それを契約として整理して文書化します。
結論から言うと、あなたが上で読んだ「設計でやること」は、すべて「基本設計」と「詳細設計」に分解されます。
違いは 「どこまで決めるか」「誰のための文書か」 です。
以下では、
## 基本設計／詳細設計の本質的な違い
先ほど説明した設計項目との対応表
あなたのDLL案件での「ちょうど良い切り分け方」
現場で通る書き方のコツ（重要）
の順で説明します。
1️⃣ 基本設計と詳細設計の違い（超要約）
観点	基本設計	詳細設計
目的	どういうシステムかを決める	どうやって実装するかを決める
視点	利用者・上位設計・差し替え	実装者・テスト
抽象度	高い（What / Why）	低い（How）
コード	書かない	書いてもよい（擬似コードOK）
API	契約として定義	そのAPIをどう実現するか
図	構成図・フロー図	クラス図・シーケンス図
一言で言うと：
基本設計＝「このDLLとは何者か」
詳細設計＝「このDLLをどう作るか」
2️⃣ 先ほどの説明との対応関係（重要）
あなたに説明した「設計でやること」を、基本設計／詳細設計に振り分けるとこうなります。
🟦 基本設計に対応するもの（成果物：基本設計書）
👉 API・契約・差し替え前提の決め事
上で説明した項目	基本設計でやること
API仕様	API一覧、引数、戻り値、呼び出し順、制約
利用シナリオ	正常系／異常系の呼び出しフロー
状態遷移	Initialize→Execute→Shutdown 等
互換性ルール	出力の同等性、許容誤差、バージョン方針
性能要件	FPS、レイテンシ、最大サイズ
スレッド制約	呼び出しスレッド条件
エラー仕様	エラーコード体系、復帰可否
外部依存	GPU、OS、ランタイム
入出力仕様	フォーマット、stride、所有権
📌 ポイント
「このDLLをどう使うか」が分かればよい
内部でCUDAかCPUかはこの段階では重要でない
他社が別DLLを作っても同じように動くことが目的
🟩 詳細設計に対応するもの（成果物：詳細設計書）
👉 実装・内部構造・品質保証の決め事
上で説明した項目	詳細設計でやること
アーキテクチャ	モジュール分割、責務
処理パイプライン	1フレームの内部処理流れ
スレッド設計	ワーカースレッド、キュー
メモリ設計	バッファ管理、再利用、寿命
GPU設計	D3D/CUDA interop、同期
エラー処理	どこで検出しどう伝播
ログ設計	ログレベル、出力内容
テスト設計	単体／結合／性能
擬似コード	Executeの内部処理など
📌 ポイント
「この設計書を渡せば実装できる」状態
実装者が自分でも、将来の自分のために書く
変更してもAPIを壊さないための安全装置
3️⃣ あなたのDLL案件での「現実的な切り分け」
あなたの案件（C#呼び出し／動画リアルタイム／差し替え）では、
**基本設計が最重要、詳細設計は“実装の安全装置”**です。
🟦 基本設計書に必ず書くべきもの（最低限）
DLLの目的・責務
API一覧（Initialize / SetParameters / Execute / Shutdown）
呼び出し順と状態遷移図
入出力フォーマット（RAW16 / Texture / stride）
スレッド制約（UIスレッド？同一スレッド？）
性能要件（30fps / 60fps）
エラーコード一覧
差し替え条件（互換性）
👉 これがないと「設計してない」と言われます
🟩 詳細設計書に書けば評価が上がるもの
内部モジュール構成図
フレーム処理のシーケンス図
GPUリソース寿命（Initialize〜Shutdown）
ワーカースレッド設計（queue / future）
Execute内部の擬似コード
テスト項目表（golden、長時間）
4️⃣ 現場で通る「書き方」の重要ルール
❌ よくあるNG
基本設計にクラス図・CUDAカーネルを書く
詳細設計にAPI仕様を書く（二重管理）
実装コードをそのまま貼る
✅ 正解
基本設計：契約・外部から見える振る舞い
詳細設計：内部構造・壊れないための工夫
5️⃣ 一文で覚える定義（これを言えればOK）
基本設計とは「このDLLが何を提供し、どう使われるかを定義する設計」であり、
詳細設計とは「その仕様を満たすために内部をどう実装するかを定義する設計」である。
次の一手（おすすめ）
次はこれをやると一気に「設計ができる人」になります。
あなたの 実際のAPI（C++ヘッダ or C# P/Invoke） を貼る
→ 基本設計書の目次＋具体記述を私がそのまま作る
もしくは
「基本設計書だけをまず通したい」
→ レビューに通る最小構成の基本設計書テンプレを実案件向けに作成
どちらに進みますか？

①基本設計書・詳細設計書（ドラフト） 
②現状設計の不足点と設計案、をまとめて出します。
## ① 基本設計書（ドラフト）
### 1. 目的・背景
本DLL（ImageProcCudaDLL）は、C#アプリからP/Invokeで呼び出され、動画フレーム（16-bit）をリアルタイム処理する画像処理DLLである。
また、他社/他実装DLLとの置き換えが想定されるため、外部API仕様（契約）を安定化し、互換性を維持することを目的とする。
### 2. 対象範囲（本DLLが提供する機能）
D3D11リソース（Texture2D or Buffer）をCUDAに登録し、インプレース処理を実行
パラメータ（WL/WW・Sobel・Blur・Invert・Threshold 等）の設定
CPU↔GPUの入出力支援（UploadRaw16 / ReadbackRaw16）
IOリソース生成補助（CreateIoSharedHandle / CreateIoBuffer）
### 3. 外部インターフェース（C API / P/Invoke境界）
ヘッダ：ImageProcApi.h
#### 3.1 API一覧（状態遷移）
初期化
IPC_Init(gpuId, ioSharedHandle)：共有Textureハンドル経由でIO TextureをOpenしCUDA登録
IPC_InitWithIoBuffer(gpuId, ioBufferPtr)：ID3D11Bufferポインタ経由でCUDA登録
設定
IPC_SetParams(const IPC_Params* p)
実行
IPC_Execute()
終了
IPC_Shutdown()
入出力
IPC_UploadRaw16(src, srcBytes)（Textureパス）
IPC_UploadRaw16ToBuffer(src, srcBytes, width, height)（Bufferパス）
IPC_ReadbackRaw16(dst, dstBytes)（Textureパス）
IPC_ReadbackRaw16FromBuffer(dst, dstBytes)（Bufferパス）
診断
IPC_GetLastHr(), IPC_GetLastErr()
リソース生成補助
IPC_CreateIoSharedHandle(gpuId, w, h) / IPC_DestroyIoSharedHandle(handle)
IPC_CreateIoBuffer(gpuId, w, h) / IPC_ReleaseD3D11Resource(ptr)
#### 3.2 呼び出し順序（基本フロー）
IPC_Init または IPC_InitWithIoBuffer
（任意）IPC_SetParams
フレームごとに IPC_Upload... → IPC_Execute → IPC_Readback...（またはIOリソースを直接利用）
IPC_Shutdown
#### 3.3 スレッドモデル（外部仕様）
外部からの呼び出しスレッドは任意（UIスレッドでも可）
DLL内部は **GpuWorker（単一ワーカースレッド）**上でD3D/CUDAを直列実行する
同時呼び出しは可能だが、内部では直列化される（同時実行はしない）
#### 3.4 入出力データ仕様（契約）
Texture IO：DXGI_FORMAT_R16_UINT の ID3D11Texture2D を前提
Buffer IO：ID3D11Buffer は w*h*2 バイト以上の容量が必要
Upload/Readback：
RAW16（Little Endian想定、1pixel=uint16）
必要バイト数：w*h*2
#### 3.5 パラメータ仕様（IPC_Params）
sizeBytes は sizeof(IPC_Params) 以上
version は 1
window, level, enableEdge
reserved[0..]：
[0] blur on/off
[1] invert on/off
[2] threshold on/off
[3] thresholdValue (0..65535)
注意：現状コードでは width/height の扱いが経路ごとに揺れている（後述）。基本設計としては どのAPIが w/h を決定するのかを契約として固定する必要がある。
#### 3.6 戻り値・エラー仕様（契約）
IPC_Result（IPC_OK, IPC_ERR_INVALIDARG, IPC_ERR_NOT_INITIALIZED, IPC_ERR_INTERNAL, IPC_ERR_INVALID_STATE …）
ただし現状は「-1300-cr」等の負値も返るため、契約としては統一が必要（後述の設計案）
#### 3.7 性能要件（ここは“基本設計”に必須）
（数値はプロジェクト要件に合わせて埋める）
例：512x512@30fps で IPC_Execute 平均 X ms 以下、p99 Y ms 以下
Execute内での不要な同期（例：cudaDeviceSynchronize）を避ける方針
1フレーム処理の計測方法（DLL内計測／呼び出し側計測）
4. 非機能要件
互換性：API/ABI維持、構造体拡張ルール、バージョニング
安定性：例外をDLL境界に漏らさない
リソース：GPUメモリリーク無し、D3D/CUDA登録解除の保証
## 詳細設計書（ドラフト）
### 1. 内部アーキテクチャ
主要モジュール：
ImageProcApi.cpp
外部C API（IPC_*）の窓口
GpuWorker による直列実行
IpcContext によるD3D/CUDAリソース保持
staging（Upload/Readback）確保とCopy
CudaInterop.cu
CUDA-D3D11 interop ラッパ
カーネル実装（WLWW / Sobel / Blur / Invert / Threshold）
中間バッファ（MID）キャッシュ（g_midArr / g_midBuf）
### 2. スレッド設計（GpuWorker）
SubmitAndWait(fn) でタスクをキュー投入し、ワーカースレッドで実行
D3D/CUDA API はワーカースレッド以外から触らない
IPC_Shutdown でリソース解放後、ワーカ停止
設計上の注意（現状実装のままだと起きやすい問題）：
Stop() はキュー内処理を完了してから終了するが、呼び出し側が Shutdown を多重に呼ぶ場合の冪等性・競合は設計で明記する必要あり
### 3. コンテキスト設計（IpcContext）
保持する状態：
D3D11 device/context
IOリソース（ioTex or ioBuf）
stagingリソース（upload/readback）
CUDA登録ハンドル（cudaGraphicsResource* cudaIO）
画像サイズ w/h
パラメータ IPC_Params params
initialized
ライフサイクル：
Init系で g_ctx を生成・初期化
Shutdownで Reset() → g_ctx破棄
### 4. IOパス別の処理詳細
#### 4.1 Texture(IO共有ハンドル)パス
IPC_Init：
GPUアダプタ選択してD3D11 device/context作成
OpenSharedResource(handle) で ID3D11Texture2D を取得
Formatチェック（R16_UINT）
CudaSetDeviceSafe(gpuId)
CudaRegisterD3D11Texture(ioTex)
default params をセット、initialized=true
IPC_Execute：
CudaMapGetArrayMapped(cudaIO) で cudaArray を取得（map）
CudaProcessArray_R16_Inplace() 実行
CudaUnmapResource(cudaIO)（unmap）
Upload/Readback：
Upload：staging texture を Map(write) → CopyResource(staging→ioTex)
Readback：CopyResource(ioTex→staging) → Map(read) → CPUコピー
#### 4.2 Buffer(IOバッファ)パス
IPC_InitWithIoBuffer：
ioBufferPtr を ID3D11Buffer にQueryInterface
buf->GetDevice() で device、GetImmediateContext() で context
CudaRegisterD3D11Buffer(ioBuf)
default params、initialized=true
IPC_Execute：
CudaMapGetPointerMapped() で devPtr と mappedBytes を取得（map）
required = w*h*2 を満たすかチェック
CudaProcessBuffer_R16_Inplace() 実行
CudaUnmapResource()（unmap）
Upload/Readback：
staging buffer を Map→memcpy→CopyResource(staging→ioBuf)
CopyResource(ioBuf→staging)→Map→memcpy
### 5. CUDA側詳細（CudaInterop.cu）
Interop：
register / map / get array or pointer / unmap / unregister
処理パイプライン：
Stage1：WL/WW or Sobel（必ず実行）
Stage2：Blur（任意）
Stage3：Invert（任意）
Stage4：Threshold（任意）
最終結果をIO側に戻す（必要ならCopy）
中間バッファキャッシュ：
Textureパス：g_midArr（cudaArray）
Bufferパス：g_midBuf（device pointer）
CudaReleaseCache() で解放
### 6. エラー処理詳細（現状）
IPC層：IPC_Result と、負の独自コード（-1300-cr など）が混在
診断：IPC_GetLastHr / IPC_GetLastErr は主に CreateIoBuffer 系で使用
### 7. テスト設計（詳細設計に含める）
正常系：
Init→SetParams→Execute→Shutdown（Texture/Buffer両方）
各reservedフラグ組み合わせ
異常系：
未InitでExecute
サイズ不一致（w/h、buffer容量不足）
dstBytes不足（Readback）
性能：
連続フレーム10000回、平均・p99・最大
長時間：
30分～数時間でリーク/速度劣化がないこと（特にMID再確保）
## ② 現状コードで「設計が不足している部分」と、設計案
ここが本題です。現状は“動く”に寄っていて、差し替え・品質保証の観点では 契約が未確定／一貫していない点がいくつかあります。重要度順に指摘＋案を出します。
### A. 「外部仕様（契約）」が未固定／揺れている
#### A-1. エラーコード体系が混在している（設計不足）
IPC_Result が定義されているのに、実装は -1300 - cudaError や -2 などの負値を返す
呼び出し側は「どの範囲が何の意味か」を知らないと扱えない
IPC_ERR_INVALIDARG と IPC_ERR_INVALID_ARG が重複していて紛らわしい
設計案（推奨）
外部契約として戻り値を IPC_Resultに統一（0=OK、非0=エラー）
CUDA/D3Dの詳細は「取得API」で返す
IPC_GetLastErrorInfo(IPC_ErrorInfo* out) のような構造体に
domain（D3D/CUDA/IPC）
code（HRESULT / cudaError）
message（固定文字列 or ring buffer）
互換性維持のため、既存負値を残すなら 範囲を規定する（例：-1000..-1999はCUDA初期化、-1300..はmap…）
#### A-2. 画像サイズ（w/h）の決定ルールが曖昧（設計不足）
Textureパス：Init時に ioTex->GetDesc() で w/h を決定
しかし IPC_SetParams が g_ctx->w = p->width; g_ctx->h = p->height; と上書きする
Bufferパス：Initではw/hが決まらず、UploadRaw16ToBufferで決めている
設計案（契約として固定すべき）
どれかに統一してください（おすすめ順）：
IOリソースの実体が真実：Textureはdesc、Bufferは「Init時にwidth/height必須」
IPC_InitWithIoBuffer(gpuId, ioBufferPtr, width, height) にする
SetParamsでサイズも確定：ただしIOリソース容量・フォーマットチェックを必須化
Upload時にサイズ確定（現状に近い）が、Execute前提条件が増え混乱しやすい
※リアルタイムDLLとしては、**“実行前提条件が少ない”**ほど品質が上がるので、(1)が強いです。
#### A-3. 共有ハンドル生成/破棄の契約が成立していない（設計不足）
IPC_CreateIoSharedHandle は内部に g_ownedSharedTex を保持し、HANDLEを返す
IPC_DestroyIoSharedHandle は実質何もしない（handleをnullptrに代入してるだけ）
つまり「誰が何を所有し、いつ解放されるか」が外から見えない
設計案
契約をどちらかに寄せる：
案1：生成補助をやめる（呼び出し側がD3Dで生成し、DLLはOpenSharedResourceのみ）
案2：生成補助を正しく“所有権付き”にする
void* IPC_CreateIoTexture(...); が opaque handle を返し
IPC_GetSharedHandle(opaque, &HANDLE) で共有ハンドル取得
IPC_DestroyIoTexture(opaque) で確実に破棄
“HANDLEだけ渡して解放もDLLで”は危険（HANDLEの寿命と実体が別問題になるため）
### B. 「スレッド／多重利用／差し替え」を想定した設計が不足
#### B-1. グローバル単一インスタンス（g_ctx）が前提（設計不足）
現状は同時に複数パイプライン（例：2動画）を処理できない
将来「別DLLに置換」や「複数ストリーム」に拡張しにくい
設計案（強く推奨）
外部APIを “コンテキストハンドル方式” にする
IPC_Handle IPC_Create(...)
IPC_SetParams(h, ...)
IPC_Execute(h)
IPC_Destroy(h)
これにより、テストもしやすく、差し替え時も契約が明確になります
（※今のAPI仕様が固定なら、内部だけでも “擬似ハンドル化” しておくと後で救われます）
#### B-2. CUDA中間バッファがグローバル（g_midArr/g_midBuf）でGPU/解像度/複数ctxに弱い（設計不足）
g_mid* は process-wide で共有されるため、
将来複数ctxや複数gpuIdで破綻しやすい
解像度切替のたびに競合・再確保が起きうる
さらに cudaDeviceSynchronize() を毎回呼び、レイテンシが伸びやすい
設計案
MIDは IpcContext にぶら下げて コンテキスト所有にする（Texture用/Buffer用を分離）
cudaStream_t を ctx に持ち、全カーネルを同一streamで発行（将来 cudaMallocAsync にも自然に繋がる）
原則 cudaDeviceSynchronize は避け、必要なら “呼び出し側が同期点を持つ” または “Fence/イベントで待つ” を契約化
### C. 「例外・診断・ログ」設計が不足（品質が出にくい）
#### C-1. GetLastErr系が部分的で、スレッド安全/原因追跡が弱い
g_lastHr/g_lastErr が主にCreateIoBuffer周辺だけ
workerスレッド実行中のD3D/CUDA失敗を呼び出し側で追いにくい
マルチスレッド呼び出し時に “最後のエラー” が混ざる設計になりやすい
設計案
エラー情報は コンテキスト単位、または **呼び出しスレッド単位（TLS）**で保持
IPC_GetLastErrorInfo() を統一し、どのAPI失敗でも同じ仕組みで追えるようにする
性能計測（平均ms/p99）やキュー長をログ出力できるフックも設計に入れる
### D. 「前提条件の明文化」が不足（レビューで刺されやすい）
今のコードには暗黙の前提が多いです。基本設計書に“契約”として明文化すると一気に強くなります。
例：
Textureは必ず R16_UINT
Bufferは w*h*2 バイト以上
IPC_SetParams の width/height は使うのか？使わないのか？
IPC_UploadRaw16 の src は呼び出しが返るまで有効であること（同期呼び出しなのでOKだが契約として書く）
IPC_Shutdown 多重呼び出しは許すか（冪等か）
設計案
“API前提条件（Preconditions）” と “API保証（Postconditions）” を各関数に付ける
これが差し替え可能性の核心です
### 仕上げ：あなたが提出物として通しやすい形
あなたの現状コードの方向性（GpuWorkerで直列化、Contextで所有）は良いです。
提出物としては、次を「設計として確定」させるだけで評価が上がります。
- 戻り値体系の統一（IPC_Result + 詳細取得API）
- w/hの決定ルールの固定（Initで確定、SetParamsはサイズを変えない等）
- IOリソース所有権の固定（CreateIoSharedHandleの扱いを整理）
- MID/streamの所有権をContextへ（将来拡張と品質の土台）
## 📘 基本設計書
ImageProcCudaDLL 基本設計書
### 1. 目的
本設計書は、C# アプリケーションから P/Invoke により呼び出される
リアルタイム画像処理 DLL（ImageProcCudaDLL） の外部仕様を定義する。
本 DLL は以下を満たすことを目的とする。
動画フレームに対するリアルタイム画像処理
CUDA を用いた GPU 高速処理
他実装 DLL との 差し替え可能性
安定した API 契約の提供
### 2. 対象範囲
本 DLL が提供する機能は以下である。
D3D11 Texture / Buffer を CUDA に登録し、インプレース画像処理を実行
画像処理パラメータ（WL/WW、Sobel、Blur、Invert、Threshold）の設定
CPU↔GPU 間の RAW16 データ入出力
IO 用 D3D11 リソース生成補助
### 3. 前提条件
OS：Windows 10 以降（x64）
GPU：CUDA 対応 GPU
API 呼び出し方式：C API（P/Invoke 想定）
画像形式：16bit Grayscale（RAW16）
### 4. 外部インターフェース仕様
#### 4.1 API 一覧
分類	API
初期化	IPC_Init / IPC_InitWithIoBuffer
設定	IPC_SetParams
実行	IPC_Execute
終了	IPC_Shutdown
入力	IPC_UploadRaw16, IPC_UploadRaw16ToBuffer
出力	IPC_ReadbackRaw16, IPC_ReadbackRaw16FromBuffer
補助	IPC_CreateIoSharedHandle, IPC_CreateIoBuffer
診断	IPC_GetLastHr, IPC_GetLastErr
#### 4.2 呼び出し順序（状態遷移）
[Uninitialized]
      |
      v
[Initialized] -- SetParams --> [Initialized]
      |
      v
[Execute] (frame loop)
      |
      v
[Shutdown]
IPC_Execute は初期化後のみ呼び出し可能
IPC_Shutdown は最終状態
#### 4.3 スレッドモデル（外部契約）
API 呼び出しスレッドは任意
DLL 内部では GPU 操作は 単一ワーカースレッドで直列化される
同時呼び出しは許容するが、内部的に順序実行される
#### 4.4 入出力データ仕様
Texture IO
ID3D11Texture2D
フォーマット：DXGI_FORMAT_R16_UINT
サイズ：Init 時に確定
Buffer IO
ID3D11Buffer
サイズ：width * height * 2 バイト以上
RAW16 データ
Little Endian
1 pixel = uint16
#### 4.5 パラメータ仕様（IPC_Params）
フィールド	説明
sizeBytes	構造体サイズ
version	構造体バージョン
window / level	WL/WW
enableEdge	Sobel
reserved[0]	Blur
reserved[1]	Invert
reserved[2]	Threshold
reserved[3]	Threshold 値
#### 4.6 エラー仕様
戻り値は IPC_Result を返す
詳細なエラー情報は取得 API で取得する
DLL 境界を例外が越えないことを保証する
#### 4.7 性能要件（例）
512×512 @ 30fps を満たすこと
IPC_Execute は不要なメモリ確保を行わない
### 5. 非機能要件
API/ABI 互換性維持
GPU/D3D リソースリーク無し
長時間実行時の性能劣化無し
📕 詳細設計書
ImageProcCudaDLL 詳細設計書
1. 内部構成
1.1 モジュール構成
モジュール	役割
ImageProcApi.cpp	外部 API 窓口
CudaInterop.cu	CUDA 処理・Interop
GpuWorker	GPU 実行直列化
IpcContext	状態・リソース管理
2. コンテキスト設計（IpcContext）
保持情報
D3D11 Device / Context
IO Texture または IO Buffer
CUDA Graphics Resource
ステージングリソース
画像サイズ（w / h）
IPC_Params
初期化状態フラグ
ライフサイクル
Init 系 API で生成
Shutdown で全リソース解放
3. スレッド設計
GpuWorker による単一ワーカースレッド
SubmitAndWait により同期実行
D3D/CUDA API はワーカースレッド内のみ実行
4. 処理フロー
4.1 Texture パス
CUDA Map
Array 取得
CUDA カーネル実行
CUDA Unmap
4.2 Buffer パス
CUDA Map
Device Pointer 取得
CUDA カーネル実行
CUDA Unmap
5. CUDA 処理パイプライン
Input
  ↓
WL/WW or Sobel
  ↓
Blur (optional)
  ↓
Invert (optional)
  ↓
Threshold (optional)
  ↓
Output
6. メモリ設計
中間バッファは Context 所有
フレーム処理中に new / malloc を行わない
サイズ変更時のみ再確保
7. エラー・ログ設計
Context 単位でエラー情報保持
IPC_GetLastErrorInfo()（将来拡張）
性能計測フックを内部に持つ
8. テスト設計
単体
API 呼び出し順序
各パラメータ組み合わせ
異常系
未初期化 Execute
サイズ不一致
バッファ容量不足
性能
連続フレーム処理
p99 / 最大時間計測
長時間
数時間連続実行
メモリ・性能劣化確認
⚠️ 設計不足点と改善設計案（重要）
1. エラーコード体系が未統一
問題
IPC_Result と負値コードが混在
設計案
戻り値は IPC_Result に統一
詳細情報は取得 API で返却
2. 画像サイズ決定ルールが曖昧
問題
Init / SetParams / Upload で w/h が上書きされる
設計案
Init 時にサイズ確定
SetParams ではサイズ変更不可
3. IO リソース所有権が不明確
問題
SharedHandle の寿命と所有者が不明
設計案
生成補助 API を廃止、または opaque handle 化
4. グローバル単一 Context 前提
問題
複数ストリーム不可
設計案
Context Handle API 方式へ拡張可能な内部構造
5. CUDA 中間バッファがグローバル
問題
複数解像度・将来拡張に弱い
設計案
Context 所有へ移行
cudaStream 統一管理
6. 前提条件の明文化不足
設計案
各 API に Precondition / Postcondition を明記
✅ まとめ
基本設計：外部契約・差し替え可能性の保証
詳細設計：実装を壊れにくくする内部構造定義
本設計により、品質・保守性・拡張性を担保する
必要であれば次にできること：
APIごとの Pre/Post 条件表
エラーコード表（正式版）
Context Handle API への改訂案
設計レビューで突っ込まれやすい点の想定Q&A
どこまで出力するか、指示してください。
①「DLL 境界を例外が越えないことを保証」とは？
②「API/ABI 互換性維持」とは？
③「GPU/D3D リソースリーク無し」はどうやって実現する？
④エラーコード体系（エラーコード表（正式版）？）の案をください。
⑤APIごとの Pre/Post 条件表
思考時間: 2 ～ 3 秒
以下、質問①〜⑤にまとめて答えます。
（※なお、こちら側の環境では「以前アップされた一部ファイルが期限切れ」と出ることがあります。今回の回答は、あなたのDLL一般要件＋これまで見えている範囲（IPC_*系API）に整合する形で作っています。もし 最新版の ImageProcApi.h/.cpp を再アップできるなら、Pre/Post表やエラーコード表を“完全に現状コードに一致”させて更新できます。）
①「DLL 境界を例外が越えないことを保証」とは？
C++側で throw が発生して、その例外が C API（extern "C"）の境界を越えて C#（P/Invoke）まで飛び出すことを防ぐ、という意味です。
C#→P/Invoke→C DLL は「例外の呼び出し規約」が一致しません。
C++例外がDLL外へ漏れると 即クラッシュ/未定義動作になりやすいです。
保証するとは、設計としてこうすること：
すべての exported API（IPC_*）の最上位で必ず例外を捕捉し、戻り値（IPC_Result）に変換する。
例外内容は IPC_GetLastErrorInfo() のような診断APIで参照できるようにする。
典型パターン（方針）
extern "C" IPC_Result IPC_Execute() の中で
try { … } catch(...) { lastError=…; return IPC_ERR_INTERNAL; }
②「API/ABI 互換性維持」とは？
API互換
関数名・引数・呼び出し順・意味が変わらないこと。
例：IPC_Execute() の意味（同期/非同期、入出力、前提条件）が変わるとAPI互換が壊れます。
ABI互換
バイナリ互換です。再コンパイルせずに差し替えられること。
P/Invoke DLLでは特に重要です。
ABI互換で注意すべき典型：
exported関数の シグネチャ（引数・戻り値・calling convention） を変えない
構造体（IPC_Params）の サイズ・配置 を壊さない
フィールド追加は「末尾に追加」＋「sizeBytes/versionで判定」が定石
32/64bit、アラインメント、#pragma pack の影響
設計としてやること
IPC_Params に sizeBytes と version を必ず持たせる（あなたの方向性は正しい）
「構造体拡張ルール」を基本設計に明記する
例：v1のアプリがv2 DLLを使っても動く、等
③「GPU/D3D リソースリーク無し」はどうやって実現する？
結論：所有権（誰が持つか）と寿命（いつ解放か）を設計で固定し、RAIIで実装するです。
（あなたが最近やっている “RAII化” はまさにここに効きます）
まず「リーク」とは
ID3D11Texture2D / ID3D11Buffer / ID3D11Device など COM参照が残って解放されない
cudaGraphicsResource* を unregister し忘れる
cudaMalloc / cudaMallocAsync したバッファを cudaFree し忘れる
map/unmap の不整合（mapしたまま例外で帰る、など）
実現の基本設計（ルール化）
IpcContextが所有するものを列挙し、Shutdownで必ず解放する
exported APIは 必ずShutdown/Resetで後始末できるよう冪等にする（2回呼ばれても安全）
すべての「取得→解放」が対になっていること（設計段階で表にする）
実装の定石（詳細設計に書くべき）
D3D COMは ComPtr<>（参照カウントを自動管理）
CUDA登録はRAIIラッパで管理
ctor/register
dtor/unregister
map/unmap もスコープガード化
mapしたら必ずunmap（例外でもunmap）
テストで保証する（重要）
長時間テスト（数時間の連続 Execute）でメモリ使用量が増えない
D3D debug layer / PIX / Nsight などでリーク検出
Init→Shutdown を1000回繰り返して増えないこと
④ エラーコード体系（正式版）の案
あなたのDLL用途（リアルタイム・P/Invoke・差し替え）だと、戻り値は 小さい固定 enum にして、詳細（HRESULT/cudaError/文字列）は別APIで取るのが一番運用が楽です。
4.1 IPC_Result（戻り値）案（安定契約）
typedef enum IPC_Result {
    IPC_OK = 0,
    // 呼び出しミス（契約違反）
    IPC_E_INVALID_ARG      = 1,   // NULL, size不足, 範囲外など
    IPC_E_INVALID_STATE    = 2,   // 未初期化でExecute等
    IPC_E_NOT_INITIALIZED  = 3,   // 初期化されていない
    IPC_E_NOT_SUPPORTED    = 4,   // GPU/format等が非対応
    IPC_E_BUFFER_TOO_SMALL = 5,   // dstBytes不足など
    // 外部API失敗（詳細は last error info で）
    IPC_E_D3D_FAIL         = 100,
    IPC_E_CUDA_FAIL        = 101,
    // 内部
    IPC_E_INTERNAL         = 200, // 想定外（例外含む）
    IPC_E_TIMEOUT          = 201  // 将来：非同期実行や待ちで使える
} IPC_Result;
4.2 追加：詳細情報取得（強く推奨）
typedef enum IPC_ErrorDomain {
    IPC_ERRDOMAIN_NONE = 0,
    IPC_ERRDOMAIN_IPC  = 1,
    IPC_ERRDOMAIN_D3D  = 2,
    IPC_ERRDOMAIN_CUDA = 3
} IPC_ErrorDomain;
typedef struct IPC_ErrorInfo {
    uint32_t sizeBytes;      // sizeof(IPC_ErrorInfo)
    uint32_t version;        // 1
    IPC_Result result;       // 最後のIPC_Result
    IPC_ErrorDomain domain;  // どこの失敗か
    int64_t nativeCode;      // HRESULT or cudaError_t 等
    char message[256];       // 短い説明（固定長でABI安定）
} IPC_ErrorInfo;
IPC_Result IPC_GetLastErrorInfo(IPC_ErrorInfo* outInfo);

こうすると「戻り値は安定」「詳細は必要時だけ」になり、C#側も扱いやすいです。
既存の IPC_GetLastHr, IPC_GetLastErr は将来的に IPC_GetLastErrorInfo に統合できます。
⑤ APIごとの Pre/Post 条件表（提出向け）
以下は “契約”として書ける形です（基本設計に載せる用）。
※ “Textureパス / Bufferパス” で分岐する項目は注記しています。
※ w/h確定ルール は推奨案（Init時確定）で書いています。現状実装が違うなら、合わせて修正するか、契約を現状に寄せる必要があります。
IPC_Init(gpuId, ioSharedHandle)
Pre（前提）
ioSharedHandle != NULL
呼び出し前に IPC_InitWithIoBuffer を含む初期化が完了していない（未初期化状態）
gpuId は有効（0..N-1）
ioSharedHandle は ID3D11Texture2D の shared handle である
Texture format は DXGI_FORMAT_R16_UINT（契約で固定する場合）
Post（保証）
成功時：Initialized状態になる
IO Texture が Open され CUDA 登録が完了している
内部で w/h が Texture desc から確定している
失敗時：未初期化状態のまま（部分初期化リソースは解放済み）
例外は漏れず、エラー情報は last error に保存される
IPC_InitWithIoBuffer(gpuId, ioBufferPtr /*, width, height 推奨 */)
Pre
ioBufferPtr != NULL
未初期化状態
（推奨契約）width>0 && height>0 を Init 引数で受け取る
現状が違う場合：少なくとも Execute 前に w/h が確定していること
ioBuffer は width*height*2 以上
Post
成功時：Initialized状態
IO Buffer が CUDA 登録済み
w/h が確定している
失敗時：未初期化へロールバック
IPC_SetParams(const IPC_Params* p)
Pre
Initialized状態
p != NULL
p->sizeBytes >= sizeof(IPC_Params_v1)（versionに応じて）
p->version が対応範囲
window/level の範囲が妥当（例：window>0 など）
（推奨契約）SetParamsでは w/h を変更しない
サイズ変更は別API（Reinit）とする
Post
成功時：以降の Execute に新パラメータが反映される
失敗時：パラメータは変更されない（or 変更はロールバック）※どちらか明記
エラー詳細は last error に保存
IPC_Execute()
Pre
Initialized状態
w/h が確定している（>0）
Textureパス：cuda resource が登録済み
Bufferパス：ioBuffer容量が w*h*2 以上
Post
成功時：IOリソース上の画像が処理後画像に更新される
map/unmapは必ず整合する（内部保証）
失敗時：IOリソースの内容は「未定義」か「元のまま」かを契約で決める
推奨：失敗時は「元のまま」を目標（難しければ未定義と明記）
実行時間：性能要件の範囲内（別章）
IPC_UploadRaw16(src, srcBytes)（Textureパス）
Pre
Initialized状態（Textureパスで初期化済み）
src != NULL
srcBytes >= w*h*2
Post
成功時：IO Texture が src の内容に更新される
失敗時：IO Texture は不変（推奨）
src は呼び出し完了まで有効である必要（同期動作）
IPC_ReadbackRaw16(dst, dstBytes)（Textureパス）
Pre
Initialized状態（Textureパス）
dst != NULL
dstBytes >= w*h*2
Post
成功時：dst に IO Texture の内容がコピーされる
失敗時：dst の内容は不定（または不変）※どちらか明記
IPC_UploadRaw16ToBuffer(src, srcBytes, width, height)
Pre
Initialized状態（Bufferパス）
src != NULL
width>0 && height>0
srcBytes >= width*height*2
ioBuffer容量が width*height*2 以上
（契約）このAPIで w/h を確定させるなら、その旨を明記
Post
成功時：IO Buffer にコピー完了
必要なら内部 w/h を width/height に更新（契約次第）
IPC_ReadbackRaw16FromBuffer(dst, dstBytes)
Pre
Initialized状態（Bufferパス）
dst != NULL
dstBytes >= w*h*2
Post
成功時：dst に IO Buffer 内容がコピーされる
IPC_Shutdown()
Pre
なし（冪等を推奨：InitializedでもUninitializedでも呼べる）
Post
全D3D/CUDAリソースが解放される
ワーカースレッド停止
状態は未初期化へ
2回呼ばれても安全（推奨）
IPC_CreateIoSharedHandle(gpuId, w, h)
Pre
w>0 && h>0
GPU/D3D11が利用可能
Post
成功時：shared handle を返す
（要設計）誰が実体を所有し、Destroyで何が解放されるかを明記必須
IPC_DestroyIoSharedHandle(handle)
Pre
handleが有効（契約上）
Post
（要設計）関連する実体が解放される
現状コードが「何もしない」に近いなら、契約を修正するか実装を修正する
IPC_CreateIoBuffer(gpuId, w, h)
Pre
w>0 && h>0
Post
成功時：ID3D11Buffer* を返す（呼び出し側が Release する or IPC_ReleaseD3D11Resource を使う）
IPC_ReleaseD3D11Resource(ptr)
Pre
ptrはD3D11 COMオブジェクトのIUnknown*
Post
Releaseされ参照カウントが減る
追加：この2つは設計書に書くと強い（おすすめ）
エラー取得APIを統一（GetLastHr/GetLastErrを置き換え可能に）
w/h確定ルールの固定（Initで確定、SetParamsで変えない、など）
必要なら、あなたが出した 基本設計書 / 詳細設計書 に、いまの回答（④エラー表＋⑤Pre/Post表）を「そのまま章として差し込んだ完成版」に整形して出し直します。
その場合、現在の ImageProcApi.h を再アップしてもらえると、関数シグネチャ・reservedの意味・現状挙動に 完全一致させた版にできます。