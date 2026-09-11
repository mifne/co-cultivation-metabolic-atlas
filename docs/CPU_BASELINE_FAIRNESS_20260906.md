# CPUを不当に遅くしない比較条件

2026-09-06。現行コードと保存済み測定の監査。性能測定は本監査では実行していない。
目的は、GPUが「CPU 1 workerだけ」ではなく、この計算機で適切に最適化したCPU構成を
上回るかを検証することである。元LP、目的関数、精度基準は変更しない。

## 現時点で言えること

`results/pf_ipm_cpu_maxmin32_speed_recheck_20260906.json`を再確認した。
対象は`pf_lp_trace_dev32x41_20260905`のstep 1 / maxmin、異なる32環境、各3回である。

| 比較対象 | 中央値 | 実際の測定範囲 |
|---|---:|---|
| CPU 1 worker | 4.220701秒 | 入力検証・fresh model作成・HiGHS solve・backend元LP認証 |
| CPU 4 workers | 1.239123秒 | 同上。各LPのHiGHSは1 thread |
| GPU batch 32 | 3.572876秒 | GPU solve API。constructor/setupは含まない |

CPU 192回の実optimizer呼出しは全て元LP基準に合格し、数値再試行は0回。
GPU値は`docs/GPU_BATCH_SPEED_RESULTS_20260906.md`の3回集計である。
GPUはCPU 1 workerの当該範囲より速いが、CPU 4 workersより約2.88倍遅い。
GPU setupには別途8.54～9.10秒かかるため、これはcold総時間のGPU優位ではない。
CPU 4 workersがこのCPUの最適並列度である証拠も、PPO全体が速くなる証拠もない。

## 実装を点検した結果

| 項目 | 現状と公平性上の扱い |
|---|---|
| CPU並列度 | `RepeatedCpuLP`本体は任意の正のworker数に対応。測定wrapperだけが1/4に制限されていた。今回1/2/4/8/10/16へ拡張した。既定1/4は過去コマンドの再現性のため維持する。 |
| CPU機種と利用可能CPU | Windows `Win32_Processor`でi7-12650H、10 cores / 16 logical processorsを確認。WSLには16 online CPU、確認用process affinityは0–15。WSL `lscpu`は8 cores × 2 threadsという仮想トポロジを表示するため、それを物理CPU構成と同一視しない。 |
| LP内の並列度 | `_options`が`threads=1, parallel=off, solver=simplex, simplex_strategy=1`を指定。環境間並列の比較では妥当な一構成だが、CPUで可能な全solver設定を探索した意味ではない。 |
| CPU永続化 | 環境ID・stage・行列形状・等式数ごとにモデルを保持。構造不変なら係数/RHS/境界/目的だけ更新する。前回合格したbasisを保持し、有効なnative basisを不必要にresetしない。 |
| 現行fresh測定 | 各repeatは新しいserviceで1回solve。`reuse_basis=True`でもこのcold値には前時刻からのbasis再利用効果はない。CPUにwarm-start機能がないという解釈は誤り。 |
| 同一入力hot repeat | wrapperに実装済みで別欄に保存。全く同じLPの再solveであり、変化するdFBAの代理にはならない。GPU同一状態再実行とだけ比較する。 |
| exchangeの構造変化 | backendは`exchange_support_updates=True`かつ`n_fluxes`指定なら、監査済みperformance-floor行に限って非ゼロsupportの変化を更新しbasisを再利用できる。既定はFalse。cold測定では差を生じないが、changing-state比較では重要。 |
| 既存rollout側の差 | `benchmark_compact_gpu.py`はdictionary分岐だけに`cpu_exchange_support_updates`を渡し、persistent分岐の`RepeatedCpuLP`には渡さない。`benchmark_temporal_lp.py`も`RepeatedCpuLP(args.workers)`のみ。現状のままGPU側の構造再利用と比較すると、CPU側だけ不要に再構築する条件が残り得る。 |
| CPU前処理 | HiGHSのpresolveを禁止していない。既定optionsのまま内部presolveを利用する。GPU専用に用意したforest/zero-face/exact-equality等のinput-only縮約はCPU wrapperには適用していない。CPUに同じ縮約を使わせた場合も候補に含める。 |
| Python overhead | CPUはthread poolでrequest検証・CSR組立・変更点検査・backend認証も行う。GPU計算に対してCPUだけのPython経費をsolve本体と混同しない。thread/process構成の差は実測前に断定しない。 |
| 出力と精度 | CPUの戻り値だけでなく内部modelから得たdualと戻りxを再検査する。成功flagだけでは合格にしない。実呼出し・再試行も数える。 |
| 参照解混入 | input loaderは保存reference x/yを読まない。現在CPUが解いたvectorをGPU入力に渡す経路もない。この分離を維持する。 |

CPU/GPU共通の既存基準は、有限値、元LP primal residual ≤ 1e-5、dual violation ≤ 1e-7、
relative KKT gap ≤ 1e-7である。CPU baseline backendの基準は今回変更していない。
GPU作業緩和・second forestには追加のdirect-dual判定もある。最終比較では、その追加診断を
両側の出力に同じ式で適用して記録する。CPUだけの閾値強化や、GPUだけの判定省略はしない。
通常FP64のdirect-dual診断は区間演算による厳密保証ではない。

## 測定計画

### 1. 同一32入力でCPU worker数を選ぶ

rootがGPU/CPU他ジョブを止めた状態を確認し、以下を実行する。
出力は新規ファイルとし、過去値を上書きしない。

```bash
CUPY_ACCELERATORS= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-cuopt-26.8/bin/python scripts/benchmark_ipm_cpu_inputs.py \
  --trace results/pf_lp_trace_dev32x41_20260905 --stage maxmin --step 1 \
  --batch 32 --workers 1 2 4 8 10 16 --repeats 3 \
  --output results/pf_ipm_cpu_worker_sweep_maxmin32_20260906.json
```

worker別中央値を比較し、最速の単発試行をCPU代表値にしない。全候補を開示する。
CPU最良構成を選んだ後、別試行でGPUと再測定する。将来はmaxminだけでなくaggregate / exchange、
別state・別seedでも確認し、GPUに都合のよいstageだけでCPU超えを主張しない。
物理core数、WSL affinity/CPU割当、電源状態、温度による周波数変化、同時ジョブを記録する。
最初から16 workersが最良と仮定しない。

### 2. 比較範囲をそろえる

次の3欄を混ぜずに報告する。

1. cold総時間: 入力準備、構造縮約/証明、CPU/GPU workspace作成、転送、solve、元座標復元、認証、終了。
2. service時間: 両側ともworkspaceを既に保持し、現在の数値入力更新から認証済み出力が利用可能になるまで。
3. kernel/solver内訳: factor、Newton/Krylov、認証等。これは全体速度の代わりにしない。

現CPUの`solve_batch_wall_seconds`はmodel作成込み、GPU3.57秒はconstructor除外である。
現行ラベルはこの差を示すが、最終的なGPU優位の主張には上記の対称な比較を追加する必要がある。
独立した検証の時間も両側同じ扱いで別記し、GPU同期完了前でtimerを止めない。

### 3. CPUも実用的なwarm再最適化を使う

環境ごとの異なる状態列を、安定したenvironment IDとstage IDで時間順に投入する。
CPUは前時刻の認証済みbasis、GPUは前時刻の認証済みGPU状態だけを使う。
現在/未来CPU解や同一入力再solveをGPUのcold proposalに使わない。

CPU persistent比較に監査済み`exchange_support_updates`経路を接続し、off/on両方を診断する。
全LPを元入力hashと元精度基準で検証し、model rebuild数、basis reuse数、係数変更数、
実simplex iteration数、数値retry数を出す。更新失敗時の安全な再構築時間も計上する。
GPUの前処理cacheに相当するCPU model/basis再利用をわざと無効にしない。

### 4. CPUにも同じinput-only縮約を許す

元LP HiGHSと、GPUで採用した同じ縮約後LPをHiGHSで解く経路を比較する。
縮約後だけのcertificateでは不十分で、primal/dualを同じ写像で元LPに戻し、全元行で検査する。
縮約のcold費用・cacheの検証費用・dual lift・元LP認証も含める。
厳密同値縮約とbounded working relaxationを区別し、後者は近似作業系と明記する。
LPデータを変えた比較であっても同一の最終精度要求を維持し、元LP HiGHSを残す。
CPUが共通の縮約で速くなれば、その速い値をGPUが超えるべき基準として使う。

### 5. 最終性能主張の条件

CPU最良候補を固定後、CPU/GPUの実行順を交互にして各5回以上の独立repeatを行う。
現wrapperはworkers順にまとめて実行するため、workerスイープは構成選定用であり、最終AB/BA検証の代用ではない。
全試行、中央値、分散、同じ条件での対応する速度比を示す。失敗・未完了を高速な完了に数えない。
必要ならthreadによるhost組立が律速かを計測し、persistent process構成や別のHiGHS solver設定も候補にする。
「CPUの理論最速」ではなく「明示した実用的CPU構成群の最良値を上回った」と範囲を限定する。

最終的にはmaxmin → aggregate → exchangeの因果的依存とdFBA状態更新を含め、
同じactions/seeds/physical horizonで両側をそれぞれ閉ループ実行する。
3段LPの全合格、軌道・終点・制約違反の基準、PPO用サンプル生成のwall timeを確認して初めて、
本来の強化学習用計算が高速化したと判断する。

## 今回の実装・検証

- `scripts/benchmark_ipm_cpu_inputs.py`: `CPU_WORKER_CHOICES=(1,2,4,8,10,16)`をCLI/関数共通で使用。
- `tests/test_ipm_cpu_benchmark.py`: 全worker選択のfake service試験、許可外worker拒否を追加。
- `pytest -q tests/test_ipm_cpu_benchmark.py`: 18 passed（2.07秒）。小さいCPU unitのみで、実GEM性能測定ではない。
- 最初のworker選択拡張時点ではCPU backendは未変更。その後、下記のnative owner-thread安全性修正を
  `src/cpu_repeated_lp.py`へ実装した。元LP、CPU/GPU精度閾値、既存結果は引き続き変更していない。

監査参照: `src/cpu_repeated_lp.py`、`scripts/benchmark_ipm_cpu_inputs.py`、
`scripts/benchmark_compact_gpu.py`、`scripts/benchmark_temporal_lp.py`、
`src/lp_exact_equalities.py`、`src/lp_direct_dual_audit.py`、上記測定JSON。

## 追記: CPU native modelのthread所有を固定

rootのworkerスイープでは、1 worker / hot repeatありの途中でnative segfaultが発生し、
最終JSONが作成されなかった。この不完全実行をCPU性能値やGPU勝利として採用しない。
malloc/free付近での異常という観測だけでは、発端となる書込み箇所を特定できない。

read-only監査で、旧backendには次の跨ぎがあった。

- モデル作成/solveはThreadPoolExecutor worker、独立検証のgetSolutionと最終破棄はmain。
- 同じenvironment IDでも、次のbatchで同じworkerに割り当てられる保証はなかった。
- closeはworkerをshutdownしてからmodelsをclearしていた。

一方、HiGHS 1.14のPython bindingはモデル配列setterで所有vectorを作り、
passModelもモデルを値として受け取る。同一入力hot repeatでは変更係数/RHS/境界setterが
そもそも呼ばれない。このため「一時NumPy配列が消えたから」とする説明は現時点で根拠が弱い。
参照: [HiGHS 1.14 bindings](https://github.com/ERGO-Code/HiGHS/blob/v1.14.0/highs/highs_bindings.cpp)、
[モデル入力](https://github.com/ERGO-Code/HiGHS/blob/v1.14.0/highs/lp_data/Highs.cpp)。

HiGHS 1.14のexecutor handleはthread-localで、thread終了時にもdisposeされる。
したがってmain側のresetGlobalSchedulerだけでworker側の寿命を制御できるとは仮定しない。
参照: [TaskExecutor宣言](https://github.com/ERGO-Code/HiGHS/blob/v1.14.0/highs/parallel/HighsTaskExecutor.h)、
[dispose実装](https://github.com/ERGO-Code/HiGHS/blob/v1.14.0/highs/parallel/HighsTaskExecutor.cpp)。

安全性修正として、各環境を固定の1-thread laneへ割り当て、native Highsの作成→更新→run→
出力コピー→disableCallbacks/clear→破棄を同じownerで完結させた。
ownerが生存している間にnativeモデルを解放し、その後laneをshutdownする。
非同期互換経路からの単独_solveも同じlaneへrouteする。batchの1件が例外でも全投入済みjobをjoinする。

通常のbenchmarkはresult.solution_snapshotに含まれるowned NumPy x/y/col-dualを使い、
main threadからnative getSolutionを呼ばない。既存コードのmodels['solver']はraw Highsではなく
安全なproxyとし、外側のgetSolution/getBasis/getOptions/getInfoはowner上で読んだhost snapshotを返す。
外側からのnative変更は拒否する。legacy models.clearによるresetもowner disposalへ接続した。
warm basis、exchange更新、solverの選択/threads/既定精度・元LP認証は変更していない。

検証: owner lifecycle fake testsと既存CPU数値/辞書/交換行/例外回復/temporal/speculative試験の
9ファイルで101 passed（1.77秒）。create/update/solve/read/clear/destroyが全て同じthread IDであること、
環境順の入替・subset・reset・作成失敗・cleanup失敗時の解放、snapshotの独立所有を検査した。
実GEM性能測定やsegfault再現試験ではない。native crash原因の確定・完全解消を主張せず、
rootによる独立processでの再測定と、必要ならnative stack採取で検証を続ける。

### 再検証: owner固定だけではnative異常を解消できていない

owner固定版を使用した実GEMのworkerスイープも、1 workerのcold試行3回が元LP認証に
全合格した後、`malloc(): invalid next size (unsorted)`で異常終了した。
試行中のcold時間は4.4566 / 4.3406 / 4.5645秒だったが、processの終了コードは1で
最終JSONは保存されていない。これらを正常完了した性能サンプルに採用しない。
thread所有の固定は安全性上の改善であり、今回のnative heap異常の修正完了ではない。
エラーが表示されたallocation/freeの位置は、先行する破損書込みの位置とは限らない。

小規模のnative再現試験では、32個の1変数LPと32個の2変数LPをそれぞれ別processで、
workers 1/2/4/8/10/16、3 repeats、cold + 同一入力hot 1回として実行した。
各processで1,152回、合計2,304回の最適化が認証を通過し、両processは終了コード0だった。
これは小LPでは再現しなかったという診断結果に限られ、実GEMでの安定性や性能の証明ではない。
現時点でCSRのnnz、stride、index dtype、NumPy所有権に原因を確定する証拠はない。

### 独立processでのcold比較とnative version診断

暫定的なcold比較では、worker構成ごとに`--repeats 1 --hot-repeats 0`を新しいprocessで
実行し、32件全ての元LP認証、結果JSONの保存、process終了コード0を確認する。
rootによる初期の独立試行はworkers 1/2/4/8/10/16の順に
4.6423 / 2.4970 / 1.5031 / 0.9425 / 0.8782 / 0.7122秒で、各32/32合格・終了コード0だった。
これらは構成選択用の各1回の暫定値であり、反復統計やpersistent安定性の根拠にはしない。
4 workersをCPU最良値とする扱いは取り下げ、16 workersも正式な比較候補に含める。

独立process方式でsolve/service区間だけを測る場合、process起動・import・入力読込・終了の
費用を別欄に示す。GPU側にも対応するsetup込み/除外の範囲を示し、GPUだけcache済み、
CPUだけ毎回起動込みといった非対称な速度比にしない。
失敗したprocessも全試行記録に残し、成功するまで再実行して失敗を除外する運用は禁止する。
単発coldの正常終了は、状態が変わり続けるpersistent dFBA環境の正常動作を保証しない。
同一入力hot反復・変更入力warm反復のheap異常調査は別の未解決項目として残す。

rootは既存highspy 1.14.0を置換せず、1.15.1を`tmp/highspy-1.15.1-comparison`へ
`--no-deps --target`で導入した。絶対パスの`PYTHONPATH`を指定した別processで、
同じ実GEM入力・owner固定版・workers 1/2/4/8/10/16・各3 repeats・cold + 同一入力hot 1回の
スイープは完了した。1,152回の実optimizer呼出し全てが元LP認証を通過、numerical retryは0、
process終了コードは0だった。結果は
`results/pf_ipm_cpu_highs1151_owner_maxmin32_sweep_20260906.json`に保存されている。

| HiGHS 1.15.1 workers | cold solve_batch中央値（秒、n = 3） |
| ---: | ---: |
| 1 | 4.4460 |
| 2 | 2.3732 |
| 4 | 1.3933 |
| 8 | 0.9312 |
| 10 | 0.8747 |
| 16 | 0.7644 |

このスイープでは1.14.0で観測したnative異常は再現しなかった。ただしversion変更で
正確にどの原因が除かれたかは未確定であり、無期限の安定動作や変更入力warm dFBAの
検証完了とはしない。hot時間を上表のcold中央値へ混ぜていない。
比較には実際にimportされたhighspyのversion/path、NumPy/Pythonのversion、solver options、
全試行の終了コードを記録する。solver versionは結果に影響しうるため別構成として報告し、
元LPのprimal ≤ 1e-5、dual ≤ 1e-7、relative KKT gap ≤ 1e-7をどちらにも維持する。

[HiGHS 1.15.0の公式release notes](https://github.com/ERGO-Code/HiGHS/releases/tag/v1.15.0)には
関連しうる保守修正が列挙されているが、それぞれの適用範囲を区別する。

- [#3087](https://github.com/ERGO-Code/HiGHS/pull/3087): nested parallel taskのinterrupt処理中に
  例外が重なってcrashする問題への対策。現在のmalloc異常と一致したとは確認できていない。
- [#2981](https://github.com/ERGO-Code/HiGHS/pull/2981): solver状態の配列容量保持や解放不足への対策。
  メモリリークの修正であり、今回のheap破損やOOMを証明する資料ではない。
- [#3081](https://github.com/ERGO-Code/HiGHS/pull/3081): column stuffingのpostsolve正当性への修正。
  当該LPでこの経路が破損原因になった証拠はない。
- [#3025](https://github.com/ERGO-Code/HiGHS/pull/3025): thread sanitizerで検出された並列data raceへの対策。
  serial simplexを用いる本実行での再現条件との一致は未確認である。

version切替でも再現する場合は、最小の実GEM入力を保存した独立再現器とnative backtrace、
可能なら同一versionのsanitizer buildによって最初の不正アクセスを特定する。
無根拠なCSR型変更、solver精度の緩和、warm basisの一律無効化で原因を覆い隠さない。
