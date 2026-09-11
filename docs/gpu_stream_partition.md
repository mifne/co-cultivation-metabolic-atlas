# GPU Stream分割監査: 同一32 LPに対する協調実行評価

## 導入

本ドキュメントは、GPUによる同一32 LP（線形計画問題）のstream分割実行に関する読み取り監査と実装提案をまとめたものです。2026-09-06時点でのコード解析に基づき、数値アルゴリズムを変更せずに`1×32 / 2×16 / 4×8`のstream配置を比較するための最小実装案を提示します。GPU実行・速度測定は未実施で、培養閉ループ/PPO全体の性能評価ではなく、保存済み開発入力に対するGPU IPMの分割可能性に焦点を当てています。

---

## 結論 (出典: GPU_STREAM_PARTITION_AUDIT_20260906.md, 2026-09-06)

現行の数値アルゴリズムを変更せず、同一32 LPを `1×32 / 2×16 / 4×8` に分けたstream配置の比較は可能です。ただし、単にThreadPoolExecutorへ既存solverを渡す実装は避ける必要があります。
最小の安全な切り口は、**単一owner thread、shardごとの独立cuDSS handle/data/bufferとnonblocking CUDA stream、greenletによるevent付き協調実行**です。

既存の `_factor_newton` と `_factor_solve` の直後にだけyieldを挿入する、benchmark専用subclass/mixinで試せます。既存solverの有限値・元Newton残差・元LP認証やthread/stream guardは削除しません。現環境にはgreenlet 3.4.0が既に導入済みで、新規依存導入は不要です。

これはCPU並列超えを保証しません。32 LPを分けるだけでは総仕事量は増えず、小バッチ化によるlaunch回数増加、Python実行量増加、factor workspaceの複製が勝る可能性もあります。

## 現在の並列性と同期点

| 部分 | 現在の構造 | 分割で変わる点 |
|---|---|---|
| 疎行列積 | `GpuBatchedIPM` が環境ごとのE/Gをblock diagonal CSRにし、一度のSpMVへ渡す | shard単位の小さいblock diagonal CSRに分かれる。元から独立環境をまとめて計算している |
| cuDSS | 1つのsymbolic union patternと `values[batch, nnz]` をuniform batchとして渡す | shardごとにhandle/dataを持ち、数値factor/solveを別streamへ投入できる |
| Newton反復 | 各環境の前反復のx/y/z/sに依存する | 環境間の進行時刻をずらせるが、同一環境内の因果順序は消せない |
| MGS | 1環境1block。2 passと各基底列の更新は順序依存 | 同時32環境なら32block相当の仕事量は変わらない。4×8が必ず有利とは限らない |
| Givens/後退代入 | 小さいHessenberg系は1環境1thread、列/行内は逐次 | sparse triangular solveそのものとは別の処理。32環境の同時処理は可能でも、1環境の依存計算が全threadに拡散するわけではない |
| 有効lane管理 | device maskで凍結するが、factor/solveのbatchサイズ自体は固定 | 小shardが全環境終了すればそのshardを止められる。1×32の遅い環境に引きずられる無駄が減る場合がある |

### 確認したソース

- `src/gpu_sparse_factor.py`: constructorがthread/device/current-streamを固定し、`_context()`で毎回検査。`cudssSetStream`を使用。analysisはconstructor内。factor/solveはprivate bufferを更新し、closeはbound streamをdrain。
- `src/gpu_batched_ipm.py`: `block_diag`によるE/G・元LP組立、`uniform_kkt_pattern`、`_factor_solve`。
- `src/gpu_globalized_ipm.py`: outer loopのactive checks、factor後と更新後のstream synchronize、retry/fallback分岐、各checkpointのhost certificate、終了時history downloads。
- `src/gpu_newton_krylov.py`: opt-inのdeferred lane checksを加えてもloop境界とnegative-error checksは同期を維持。
- `src/gpu_krylov_microkernels.py`: MGS/Givensの実際のlaunch構成。
- `src/gpu_forest_ipm.py`: full-original primal/dual postsolveと元LP全行認証。受理状態はhostにも保持。
- `scripts/microbatch_comparison_support.py`: 別目的だが単一threadのghost schedulingを既に使用。

特にcuDSS数値solveは、独立行列のbatchを受け取ることと、各行列の三角依存を完全に並列化できることを同一視できません。現在のソースだけからcuDSS内部kernel占有率や依存graphのどのlevelが律速かまでは断定できず、実測/traceが必要です。

### 注意点

- NVIDIA公式の現行説明は、複数host threadからのcuDSS利用についてthread safetyを保証していません。
- analysisを先に全streamで完了し、その後factor/solveをstreamへ投入する運用が案内されています。
- analysisは同期的であり、hybrid機能無効時のfactor/solveは非同期的とされます。
- [NVIDIA cuDSS General Description](https://docs.nvidia.com/cuda/cudss/general.html)
- 現ワークスペースのadapterは0.7.x ABI固定、実測記録は0.7.1であるため、このURL文書を「導入済み0.7.1でthread並列factorを保証する根拠」にはできません。
- CuPyのcurrent streamはthread/device単位であり、thread単位ではありません。
- [CuPy 14.2 Current Stream](https://docs.cupy.dev/en/stable/user_guide/basic.html#current-stream)

## 最小実装案

新しい実験用moduleとscriptだけで試す。production backendの既定値は変更しません。

1. SHA検証した32個の異なる元LP入力を一度だけ読み込み、環境IDで `i % shard_count` に分割する。全配置で同一集合・一意性・各環境の元problem SHAを照合し、CPU参照x/yは読まない。
2. main threadでnonblocking streamを1/2/4本作る。それぞれのstream context内で独立solverを**順次constructor実行**し、全analysis/setupを完了させる。GPU arrayとcuDSS addressは共有しない。
3. benchmark専用mixinの `_factor_newton` が `super()`で本来の処理を終えた直後、そのstreamにeventを記録してschedulerへyieldし、再開時に同じscalingを返す。`_factor_solve`も本来のdevice有限guard/ownership return配列まで実行後、同様にyieldする。公開API `solve`、private factor guards、元の停止判断は書き換えない。
4. schedulerは各shardのghostを取り決めて開始し、未完了eventを登録する。eventがreadyになったshardだけ、そのstream context内で再開する。readyが1つもなければ、1つのpending eventだけを待つ。device-wide synchronizationは導入しない。
5. 1shardあたり同時にpending eventは1つなので、再利用可能なdisable-timing eventを1つ持てる。eventをpendingのまま配送しない。各yieldにstream pointer・owner thread・shard IDを記録する。
6. 完了後に元環境順へ戻し、返されたfull-original x/yを独立に再認証する。finallyで各solverを元stream context内からcloseし、すべてのpending workをdrainする。1shardの例外を成功扱いにせず、他shardのclose漏れと失敗データ混入を防ぐ。

### 擬似コードの要点

```python
# A yield occurs only after original helper arithmetic has been enqueued.
class CooperativeMixin:
    def _factor_newton(self, ratio):
        scaling = super()._factor_newton(ratio)
        scheduler.yield_after_enqueue(self.shard_id)
        return scaling

    def _factor_solve(self, rhs):
        result = super()._factor_solve(rhs)
        scheduler.yield_after_enqueue(self.shard_id)
        return result

# thread share one Python thread; restore stream on EVERY resume.
with streams[shard_id]:
    outcome = workers[shard_id].switch()
```

このseamの利点は、full/condensed Krylovとretryも既存helperを通るため、各数値アルゴリズムをgeneratorに書き直さず利用できることです。残る同期点のすべてを隠せるわけではありません。event-ready後の小演算やhost　certificateで同期する間は、他shardの先行GPU処理が進む程度です。

### 追加コスト

- event記録/query、協調切替、stream context復元には追加コストがあります。
- 毎回のfactor/solveへyieldするため回数が増えます。`factor-only` と `factor-and-solve` を同じ実装で切り替えて比較します。
- 1×32のcontrolもschedulerを通す測定の両方を残し、scheduler自体のcostを測ります。
- 既存solverの `factor_seconds` 等はPython wall時刻で測っており、yield中に他shardが走る時間を含むため、それらの値を合算してもkernel実行時間と呼べません。

## 比較契約

| 比較項目 | 必須条件 |
|---|---|
| 問題 | 同じ32 distinct environment、同じphysical step、同じLP stage、全problem SHA一致。padding/複製なし |
| 設定 | regularization、initialization実行、microkernels、refinement、retry、Krylov上限、check interval、reduction flagsを固定 |
| 精度 | 全32解が元LPのprimal ≤1e-5、dual violation ≤1e-7、relative KKT gap ≤1e-7、finite。追加direct-dual gateも同じにする |
| CPU | 同じ入力/精度でthreadを保持するHiGHS。workers 1/2/4/8/10/16を調査し、最も速い設定も示す。4並列だけをCPU上限としない |
| cold | host構造変換、全constructor/analysis、全solve、postsolve、必要な転送・認証・closeを含む総時間を独立列へ。setupの償却は右にしない |
| solve-only | 全constructorとqueueのdrain完了後から、全shardのsolve APIが完了してstreamがdrainされるまで。各shard秒数の合計/最小値ではない |
| 反復 | 最低3回の単独プロセス実測を配置ごとに行い、実行順を交互化。中央値・全環境の認証結果を保存 |
| キャッシュ | kernel/runtime warmupは別途記録。GPUに同じ環境解の再投入によりiteration0化した時間をcold/新の値へ混ぜない。CPU基礎再利用も同一範囲で扱う |
| 資源 | GPU model、software version、VRAM、CPU thread設定、ジョブの有無。OOMはfailureとする |
| 失敗 | 0/32や一部失敗の停止時間を有効LP/sとして扱わない。失敗率、停止理由、未認証environmentを別途記録 |
| 達成範囲 | 単一stageの同一入力速度であり、dFBA/PPO全体のCPU超え・完全GPU内完結を主張しない |

## 実験計画

このstream experimentで優位が出なければ、その結果自体が有益である。要素の結果次第で：

- GPU launch律速 → 固定作業buffer・kernel融合・同期集約
- 三角依存/数値収束律速 → 編約と級化・反復数削減

stream数や無意味な計算を増やしてGPU使用率だけを上げる方法は採らない。

## 検証計画

### 実装前の検証

- CPU mock schedulerで実行順、準備のみ再開、stream復帰、一度だけの環境ID、例外時の全closeを確認。
- 単一CUDA toyで通常実行と2/4shardの元LP全行認証を比較。終了時間の相違、invalid lane、意図に含む他のshardの値を混入しないことを検査。

### 実装直後の検証

- 複数streamで再入しても同一thread/stream guardが有効であることを検査。共有factorのstream変更でguardを迂回しない。
- 実32環境の同一集合・完全返却を検査した後に速度比較を採用。
- GPU kernel time取得可能環境では別profile実行でoverlapを検証。WSLでtraceが不完全ならその限界を明記する。

## 構成元ファイル

- docs/GPU_STREAM_PARTITION_AUDIT_20260906.md (2026-09-06)
