# 同一32 LPのGPU stream分割監査

2026-09-06。読み取り監査と実装提案。ここではGPU実行・速度測定をしていない。
対象は保存済み開発入力に対するGPU IPMであり、培養閉ループ/PPO全体の性能ではない。

## 結論

現行の数値アルゴリズムを変更せず、同一32 LPを `1×32 / 2×16 / 4×8` に分けた
stream配置の比較は可能。ただし、単にThreadPoolExecutorへ既存solverを渡す実装は避ける。
最小の安全な切り口は、**単一owner thread、shardごとの独立cuDSS handle/data/bufferと
nonblocking CUDA stream、greenletによるevent付き協調実行**である。

既存の `_factor_newton` と `_factor_solve` の直後にだけyieldを挿入する、benchmark専用
subclass/mixinで試せる。既存solverの有限値・元Newton残差・元LP認証やthread/stream guardは
削除しない。現環境にはgreenlet 3.4.0が既に導入済みで、新規依存導入は不要。

これはCPU並列超えを保証しない。32 LPを分けるだけでは総仕事量は増えず、小バッチ化による
launch回数増加、Python実行量増加、factor workspaceの複製が勝つ可能性もある。

## 現在の並列性と同期点

| 部分 | 現在の構造 | 分割で変わる点 |
|---|---|---|
| 疎行列積 | `GpuBatchedIPM` が環境ごとのE/Gをblock diagonal CSRにし、一度のSpMVへ渡す | shard単位の小さいblock diagonal CSRに分かれる。元から独立環境をまとめて計算している |
| cuDSS | 1つのsymbolic union patternと `values[batch, nnz]` をuniform batchとして渡す | shardごとにhandle/dataを持ち、数値factor/solveを別streamへ投入できる |
| Newton反復 | 各環境の前反復のx/y/z/sに依存する | 環境間の進行時刻をずらせるが、同一環境内の因果順序は消せない |
| MGS | 1環境1block。2 passと各基底列の更新は順序依存 | 同時32環境なら32block相当の仕事量は変わらない。4×8が必ず有利とは限らない |
| Givens/後退代入 | 小さいHessenberg系は1環境1thread、列/行内は逐次 | sparse triangular solveそのものとは別の処理。32環境の同時処理は可能でも、1環境の依存計算が全threadに拡散するわけではない |
| 有効lane管理 | device maskで凍結するが、factor/solveのbatchサイズ自体は固定 | 小shardが全環境終了すればそのshardを止められる。1×32の遅い環境に引きずられる無駄が減る場合がある |

確認したソース:

- `src/gpu_sparse_factor.py`: constructorがthread/device/current-streamを固定し、`_context()`で毎回検査する。`cudssSetStream`を使用。analysisはconstructor内。factor/solveはprivate bufferを更新し、closeはbound streamをdrainする。
- `src/gpu_batched_ipm.py`: `block_diag`によるE/G・元LP組立、`uniform_kkt_pattern`、`_factor_solve`。
- `src/gpu_globalized_ipm.py`: outer loopのactive checks、factor後と更新後のstream synchronize、retry/fallback分岐、各checkpointのhost certificate、終了時history downloads。
- `src/gpu_newton_krylov.py`: opt-inのdeferred lane checksを加えてもloop境界とnegative-error検査は同期を維持する。
- `src/gpu_krylov_microkernels.py`: MGS/Givensの上記実際のlaunch構成。
- `src/gpu_forest_ipm.py`: full-original primal/dual postsolveと元LP全行認証。受理状態はhostにも保持。
- `scripts/microbatch_comparison_support.py`: 別目的だが単一threadのgreenlet schedulingを既に使用。

特にcuDSS数値solveは、独立行列のbatchを受け取ることと、各行列の三角依存を完全に並列化
できることを同一視できない。現在のソースだけからcuDSS内部kernel占有率や依存graphの
どのlevelが律速かまでは断定できず、実測/traceが必要である。

## ライブラリの制約

NVIDIA公式の現行説明は、複数host threadからのcuDSS利用についてthread safetyを保証して
いない。analysisを先に全streamで完了し、その後factor/solveをstreamへ投入する運用を案内
している。analysisは同期的であり、hybrid機能無効時のfactor/solveは非同期的とされる。
[NVIDIA cuDSS General Description](https://docs.nvidia.com/cuda/cudss/general.html)

このURLは現在の公開文書であり、閲覧時の目次には0.8.0 migration guideが含まれる。
現ワークスペースのadapterは0.7.x ABI固定、実測記録は0.7.1であるため、文書を
「導入済み0.7.1でthread並列factorを保証する根拠」とは扱わない。ここではより保守的に
すべてのcuDSS host API呼出しを同じthreadから行い、既存所有権guardをそのまま満たす。

CuPyのcurrent streamはthread/device単位であり、greenlet単位ではない。そのためschedulerが
**各greenletを再開する直前**に対応stream contextを設定する必要がある。
[CuPy 14.2 Current Stream](https://docs.cupy.dev/en/stable/user_guide/basic.html#current-stream)

nonblocking streamを明示し、legacy default streamとの意図しない順序依存を避ける。
streamを増やすだけで同時kernel実行が保証されるわけではない。
[CuPy Stream API](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html)

## 最小実装案

新しい実験用moduleとscriptだけで試す。production backendの既定値は変更しない。

1. SHA検証した32個の異なる元LP入力を一度だけ読み込み、環境IDで `i % shard_count` に分割する。
   全配置で同一集合・一意性・各環境の元problem SHAを照合し、CPU参照x/yは読まない。
2. main threadでnonblocking streamを1/2/4本作る。それぞれのstream context内で独立solverを
   **順次constructor実行**し、全analysis/setupを完了させる。GPU arrayとcuDSS objectは共有しない。
3. benchmark専用mixinの `_factor_newton` が `super()`で本来の処理を終えた直後、
   そのstreamにeventを記録してschedulerへyieldし、再開時に同じscalingを返す。
   `_factor_solve`も本来のdevice finite guard/owned return arrayまで実行後、同様にyieldする。
   公開API `solve`、private factor guards、元の停止判定は書き換えない。
4. schedulerは各shardのgreenletをround-robinで開始し、未完了eventを登録する。
   eventがreadyになったshardだけ、そのstream context内で再開する。readyが1つもなければ、
   1つのpending eventだけを待つ。device-wide synchronizationを導入しない。
5. 1shardあたり同時にpending eventは1つなので、再利用可能なdisable-timing eventを1つ持てる。
   eventをpendingのまま上書きしない。各yieldにstream pointer・owner thread・shard IDを記録する。
6. 完了後に元環境順へ戻し、返されたfull-original x/yを独立に再認証する。
   finallyで各solverを元のstream context内からcloseし、すべてのpending workをdrainする。
   1shardの例外を成功扱いにせず、他shardのclose漏れと失敗データ混入を防ぐ。

擬似コードの要点:

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

# Greenlets share one Python thread; restore stream on EVERY resume.
with streams[shard_id]:
    outcome = workers[shard_id].switch()
```

このseamの利点は、full/condensed Krylovとretryも既存helperを通るため、各数値アルゴリズムを
generatorに書き直さず利用できることである。残る同期点のすべてを隠せるわけではない。
event-ready後の小演算やhost certificateで同期する間は、他shardの先行GPU処理が進める程度である。

event記録/query、greenlet切替、stream context復元には追加コストがある。毎回のfactor/solveへ
yieldすると回数が増えるため、`factor-only` と `factor-and-solve` を同じ実装で切り替えて比較する。
1×32のcontrolもschedulerを通す測定と従来直接呼出しを両方残し、scheduler自体のcostを測る。

注意: 既存solverの `factor_seconds` 等はPython wall時計で測っている。yield中に他shardが
走る時間を含むため、協調実行時のそれらを合算したりkernel実行時間と呼んではならない。
主性能値はscheduler全体のsynchronized makespanである。event pending数は投入の重なりを
示すだけで、kernel同時実行の証明ではない。

## 比較契約

| 比較項目 | 必須条件 |
|---|---|
| 問題 | 同じ32 distinct environment、同じphysical step、同じLP stage、全problem SHA一致。padding/複製なし |
| GPU配置 | 直接1×32、協調1×32、2×16、4×8。まずmaxminで試し、通過後に後段を別表で測る |
| 設定 | regularization、initialization、microkernels、refinement、retry、Krylov上限、check interval、reduction flagsを固定 |
| 精度 | 全32解が元LPのprimal ≤1e-5、dual violation ≤1e-7、relative KKT gap ≤1e-7、finite。適用中の追加direct-dual gateも同じにする |
| CPU | 同じ入力/精度でowner threadを保持するHiGHS。workers 1/2/4/8/10/16を調査し、最も速い適格設定も示す。4並列だけをCPU上限としない |
| cold | host構造変換、全constructor/analysis、全solve、postsolve、必要な転送・認証・closeを含む総時間を独立列へ。setupの償却を仮定しない |
| solve-only | 全constructorとqueueのdrain完了後から、全shardのsolve APIが完了してstreamがdrainされるまで。各shard秒数の合計/最小値ではない |
| 反復 | 最低3回の単独プロセス実測を配置ごとに行い、実行順を交互化。中央値・全raw timings・全解の認証結果を保存 |
| キャッシュ | kernel/runtime warmupは別記。GPUに同じLP解を再投入してiteration0化した時間をcold/新状態の値へ混ぜない。CPU basis再利用も同一範囲で扱う |
| 資源 | GPU model、software version、VRAM、CPU thread設定、同時ジョブの有無。OOMは失敗であり環境数を黙って減らさない |
| 失敗 | 0/32や一部失敗の停止時間を有効LP/sとして表示しない。失敗率、残差、停止理由、未認証環境を別記 |
| 達成範囲 | 単一stageの同一入力速度であり、これだけでは3段の因果的dFBA/PPO全体のCPU超え・完全GPU内完結を主張しない |

このstream実験で優位が出なければ、その結果自体が有用である。host launch律速なら固定作業
buffer・kernel融合・同期集約を、三角依存/数値収束律速なら縮約と前処理・反復数削減を優先する。
stream数や無意味な計算を増やしてGPU使用率だけを上げる方針は採らない。

## 実装前後の検証

- CPU mock schedulerで実行順、readyのみ再開、stream復元、exactly-once環境ID、例外時全closeを確認。
- CUDA toyで1×Bの通常実行と2/4shardの元LP全行認証を比較。different finish times、invalid lane、
  deliberate exceptionを含め、他shardの値を混入しないことを検査。
- 複数streamで再入してもsame owner thread/stream guardが有効であることを検査。共有factorの
  stream差替えでguardを迂回しない。
- 実GEM 32入力の同一集合・full returned pairを検査した後にのみ速度比較を採用。
- GPU kernel timelineを取れる環境では別のprofile runで重なりを検証する。WSLでtraceが
  不完全ならその限界を明記し、kernel concurrencyを断定しない。
