# 精度を維持するdFBA実行順序の再設計（2026-09-05）

## 今回の変更

OR16、NS21、P. freudenreichiiの3種community dFBAについて、LP演算とhost側の環境更新を重ねる実行方式を追加した。各環境のmaxmin → aggregate → exchangeという目的の順序を保ち、合格したフラックスから次の状態を計算する。GPUはmaxminの候補再構成・元LP検証を環境群のバッチで処理し、未合格分をCPU HiGHSへ戻す。aggregate/exchangeは段階ごとにCPUへ直接送り、完了した環境の状態更新を進める。

```mermaid
flowchart LR
  A[各環境の状態・操作] --> B[全環境のmaxmin候補をGPUで検証]
  B -->|合格| D[maxmin全環境の結果が揃う]
  B -->|未合格| C[CPU HiGHS]
  C --> D
  D --> E[環境ごとにaggregateをCPUへ発行]
  E --> F[完了した環境のexchangeをCPUへ発行]
  F --> G[完了した環境の状態を主スレッドで更新]
  G -->|全環境の次入力が揃う| B
```

CPUのLP計算中にも別環境の主スレッド処理を進められる。COBRAとgreenletは主スレッドに留め、workerにはLP配列だけを渡す。同じ環境に未完了のCPU要求は最大1件で、HiGHSの基底は環境ID・目的段階・LP形状ごとに保持する。例外時は発行済みの計算をすべて回収して終了する。

同時に、全ステップで共通の等式行列、maxmin目的ベクトル、exchangeの補助変数配置と固定係数をテンプレート化した。biomass由来の係数、供給上限、培地RHS、live objective、前段階の性能保持制約は毎回更新する。元のLP入力96件のhashと終点は変更前後で完全一致した。

## なぜこの変更を優先したか

前回の通常32×8計算では、LPサービス外のhost処理が総時間の約30%を占めた。GPU補正を増やした試験では、補正時間の約90%がoperator準備とCUDA graph構築だった。後段のaggregate/exchangeは全候補辞書でも合格率が低く、GPU処理を増やすだけでは総時間が減らなかった。

今回、直前の厳密CPU基底を次ステップのLPで再構成する診断も行った。seed20292001–04、0.2時間刻み、step2–8の各28件でaggregate/exchangeとも合格0件だった。両段階とも27件で主実行可能性違反が生じ、最終基底の列statusも全件変化していた。したがって、この条件で無補正の時系列写像を毎回作る方針は採用しなかった。

これは基底再利用一般の否定ではない。[SurfinFBA](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007786)では将来方向に適した基底選択も扱い、[mpFBA研究](https://arxiv.org/abs/1802.02567)も代謝モデルの縮退・多重最適性を考慮している。現GPU辞書は既に低次元写像と元LP検証を使っており、候補数の拡大だけでなく基底の有効範囲・修復費用を評価する必要がある。

本実装は3種を結合した階層community LPで、菌体量に依存する行列係数も変化する。また今回の比較操作列は各seedから生成した120行×5操作の乱数列であり、学習済み方策や一定流加の軌道ではない。個別菌の目的を扱うSurfinFBAの報告倍率をそのまま移せない。今回の診断は現行の0.2時間刻み・選択基底・操作列での結果に限られ、将来方向に沿った基底選択や制御入力の区間内再利用は別途検証する余地がある。

## 比較条件と精度基準

- 32環境、8ステップ、各環境の模擬培養1.6時間（1ステップ0.2時間）。1組あたり256環境遷移、768 LP。
- 2組の非重複seed群。同じ組ではCPU/GPUおよび通常/非同期に同じ初期条件・操作列を与えた。
- CPUはHiGHS、同じoffline辞書による初期基底、前時刻基底の再利用、4 worker、各LP 1 thread。非同期化はCPU比較側にも適用した。
- GPUは環境別K4候補、maxminのみ、GPU修復0回。後段2段階はCPU直行。CPU/GPU実行順は組ごとに交互。
- オンライン総時間はhost処理、GPU候補計算、転送、CPU fallback、初回オンライン使用費用を含む。offline辞書生成・初期セットアップ・環境複製の時間は別記。
- 元LPの主残差≤1e-5、双対違反≤1e-7、相対KKT gap≤1e-7。終点PHA相対差≤1%、菌体量最大絶対差≤0.01 g/L、PHV比率絶対差≤0.01。

## 32環境・8ステップの結果

| 同じseed群 | CPU・通常待機 | CPU・非同期 | ハイブリッド・通常待機 | ハイブリッド・非同期 |
|---|---:|---:|---:|---:|
| 1 | 46.465秒 | 39.867秒 | 44.076秒 | 38.008秒 |
| 2 | 44.476秒 | 38.087秒 | 43.880秒 | 37.261秒 |

ハイブリッドの非同期化により、同じハイブリッド通常待機から13.77% / 15.08%短縮した。非同期CPUと非同期ハイブリッドの比較では4.66% / 2.17%短縮で、時間比CPU/Hybridは1.04890 / 1.02216だった。通常待機CPUから非同期ハイブリッドへ変更した総効果は18.20% / 16.22%短縮である。総効果をGPU演算だけの効果とは呼ばない。

CPU/GPU間とscheduler変更前後のPHA、菌体量、PHV比率の終点差は両組とも0だった。CPU呼出し数も通常待機と非同期で同じ683 / 679件（各768 LP）であり、計算を省略して得た短縮ではない。CPU処理は依然88.9% / 88.4%のLPを担当する。

2組は開発用比較であり、信頼区間や一般的な速度優越性は主張しない。通常待機の2組を先に測り、その後に非同期の2組を測っているため、scheduler間比較の実行時点による変動は完全には除去していない。GPU/CPUの比較順は各測定内で交互にした。

![同じseed群での非同期実行比較](../results/pf_stage_pipeline_comparison_32x8_v2_20260905.png)

図は共通のゼロ起点縦軸を使い、2組それぞれの実測値を表示する。エラーバーは付けていない。SVG版も同名で保存した。

### 非同期化後に残る費用

最後の組のハイブリッド37.261秒について、非重複の主スレッド区間はmaxminサービス5.550秒、環境処理17.583秒、CPU要求準備3.039秒、CPU完了待機10.923秒、scheduler付帯処理0.158秒だった。環境処理とCPU要求準備の間にもworkerはLPを解いている。環境処理区間はCPU負荷競合やGIL待ちの影響を含み得るため、通常待機のhost単独時間との単純比較はしない。aggregate/exchangeの非同期spanは重複するため、上記へ加算しない。

## 120ステップ検証

上記の実装を固定し、新しいseed20292701–32で32環境×120ステップ（各環境24時間の模擬培養）を完了した。CPU/GPU双方に非同期schedulerを適用した1組の連続比較である。

| 指標 | CPU | GPU/CPU hybrid |
|---|---:|---:|
| オンライン総時間 | 845.541秒（14.09分） | 898.378秒（14.97分） |
| 完走環境数 | 32 / 32 | 32 / 32 |
| CPUへのLP要求数 | 11,520 | 10,873 |
| HiGHS実行数（数値再試行込み） | 11,548 | 10,902 |

HybridはCPU比6.249%長く、CPU/Hybrid比は0.941186だった。したがって、**8ステップでの小幅なGPU優位は120ステップでは維持されなかった**。この方式を長時間計算の高速化達成やRLの既定経路への昇格とは扱わない。CPU先行1組のため、実行順・熱状態等による変動とアルゴリズム差は完全に分離できていない。

原LP証明は全23,040件（CPU/GPU各11,520要求）が合格した。終点差の最大はPHA相対2.3423e-11（百分率2.3423e-9%）、菌体量1.1458e-11 g/L、PHV比率5.0245e-12であり、既定の1% / 0.01 g/L / 0.01を維持した。これはCPU数値計算との一致であって、生物学的な予測精度の証明ではない。

GPU辞書のmaxmin採用は647 / 3,840件（16.85%）、全LP要求に対しては5.62%である。41–60ステップのmaxmin採用は約3%まで低下した。aggregate/exchangeを含む全LPの94.38%はCPUが担当しており、完全GPU内完結ではない。

| 非重複の主スレッド区間 | CPU（秒） | Hybrid（秒） |
|---|---:|---:|
| maxminサービス（fallback込み） | 69.868 | 85.824 |
| host状態処理の区間 | 301.636 | 318.059 |
| CPU要求準備 | 44.540 | 49.228 |
| CPU完了待機 | 428.186 | 442.249 |
| scheduler付帯処理 | 1.309 | 3.001 |

CPU完了待機が約半分、host状態処理区間が約35%を占める。maxminサービスの増分は15.956秒だが、総増分52.837秒はそれだけでは説明できない。workerのLP計算はhost処理中も進むため、この表をCPU/GPU使用率や独立した各kernel時間に読み替えない。setup7.264秒、CPU/GPU用64環境の構築86.602秒はオンライン時間の外に記録した。

![120ステップの連続時間・待機・辞書採用率](../results/pf_stage_pipeline_diagnostics_32x120_20260905.png)

(a)は各cycleの累積時間、(b)は非重複の主スレッド区間、(c)は20ステップ区間ごとのmaxmin辞書採用率。非同期LPのspanを重複加算せず、ハードウェア使用率と区別した。生データは`results/pf_stage_pipeline_32x120_20260905.json`、厳格な再集計と図は`results/pf_stage_pipeline_diagnostics_32x120_20260905.*`。

## 実装と検証範囲

### 追加実験: GPUからCPUへ基底を引き継ぐ

120ステップ比較後、`--cpu-basis-handoff`を追加した。GPUで直前に原LP証明に合格した辞書基底をCPU復帰時の初期候補として渡す。現在LPの係数等を反映し終えた有効な既存CPUモデルにだけ適用し、cold/rebuildでは従来初期化を維持する。基底設定が失敗したら旧基底へ復元し、復元できなければcold solveへ戻す。元LPの再求解と証明は常に実施する。

4環境×8ステップでは適用11件の反復が29.17%減ったが総時間は短縮しなかった。そこで同じsource・モデル・初期条件・操作・実行順を保ち、新規seed20293101–64の32×8・2組をoff/onで比較した。

| 指標 | 組0 off → on | 組1 off → on |
|---|---:|---:|
| Hybrid総時間（秒） | 37.804 → 38.226 | 37.771 → 37.793 |
| 同時期のCPU比較（秒） | 39.901 → 40.647 | 38.347 → 38.246 |
| maxminのCPU反復数 | 20,078 → 22,311 | 21,718 → 21,390 |
| handoff適用 / 提案数 | 37 / 55 | 42 / 53 |

CPU/GPU間およびoff/on間の全終点は完全一致、CPU要求数も679 / 687で同じだった。適用79件はすべて元LP証明合格・数値再試行0件で、setter拒否も0件だった。未適用29件はcold/rebuild扱いで従来経路を維持した。

適用79件の反復数だけでは24,166→22,388（7.36%減）だったが、48件改善・31件悪化だった。workerのsetup合計は0.613→1.219秒、solve合計は8.685→8.477秒で、setup+solveは9.298→9.696秒と増えた。これらは並列workerの経過時間合計で、pipeline wallへ加算できない。全CPU要求の反復数は194,143→196,048と増加し、最新基底への切替が後の再最適化にも影響することを示している。各LPの全入力一致を保存して比較した実験ではないため、行別差の原因を単一の操作へ断定しない。

Hybrid総時間の2組合計も75.575→76.020秒で短縮せず、**handoffは既定offの実験オプションに留める**。新しい基底であることは、次のLPに適した基底であることや、因数分解費用を回収できることを保証しない。120ステップの主結果にはこの追加変更を含めていない。

再現可能な比較集計は`results/pf_handoff_pair_analysis_32x8_20260905.json`、元データは`results/pf_handoff_off_32x8_20260905.json`と`results/pf_handoff_on_32x8_20260905.json`。`scripts/analyze_basis_handoff_pair.py`は双方の完走、元LP証明、終点から再計算した誤差、source/model/辞書/seed/orderの一致を確認する。時間差を統計的優越性とは扱わない。

### 変更箇所

- `src/community_solver.py`: 静的LP構造の再利用。
- `src/gpu_hybrid_lp.py`: GPU対象段階の明示選択とCPU直行経路。
- `src/cpu_dictionary_lp.py`: 1環境の要求準備とCPUサービスの共有所有。
- `scripts/pipelined_microbatch.py`: 非同期の実行順序制御。
- `scripts/benchmark_compact_gpu.py`: 同条件CPU/GPU比較、source snapshot、初期設定・オンライン時間の記録。
- `scripts/probe_temporal_basis_validity.py`: 軌道に影響させない前基底の有効性診断。
- `scripts/compare_pipeline_runs.py`: source・seed・設定・精度の一致を検査して比較図を生成。
- `src/cpu_repeated_lp.py`: 明示された基底handoffの適用・拒否・旧基底復元と診断。
- `scripts/summarize_pipeline_run.py` / `scripts/plot_pipeline_run.py`: 完走・元LP証明・生の終点誤差を再検査し、重複しない時間内訳を集計・描画。
- `scripts/analyze_basis_handoff_pair.py`: handoffの同条件off/on比較とprovenance記録。

120ステップ測定前の関連回帰テスト475件、その後の基底handoffと集計器の追加後は539件が合格した（実GPUを含む、全リポジトリのテスト完了を意味しない）。これはdFBA環境計算の比較で、PPO更新や学習全体の倍率ではない。新schedulerは明示オプションの比較実装であり、RLの既定経路は変更していない。

paired解析器には別途6件のテストを実施し、不完全な結果・非有限終点・証明未合格・source差・環境ID不整合を拒否することも確認した。交換分類cacheの試作は実モデルでの総費用改善が未確認で、simulatorには統合していない。

生データは`results/pf_stage_serial_32x8_20260905.json`、`results/pf_stage_pipeline_32x8_20260905.json`、比較表・図は`results/pf_stage_pipeline_comparison_32x8_v2_20260905.*`。各測定の`.sources/`に使用コードを保存している。

### 再現コマンド（WSL / Linux）

作業ディレクトリは`/home/reiya/co-cultivation`。既存結果の上書きを防ぐため、再測定時は未使用の`--output`名を指定する。モデル・辞書のfingerprint検証に失敗する場合、検証を無効化せず辞書とモデルの対応を確認する。

```bash
CUPY_ACCELERATORS= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-cuopt-26.8/bin/python scripts/benchmark_compact_gpu.py \
  --bank results/pf_compact120_v3_20260904 \
  --repair-operators results/pf_compact120_repair_20260904 \
  --output results/pf_stage_pipeline_32x120_reproduction.json \
  --steps 120 --environments 32 --repeats 1 --cpu-workers 4 \
  --seed 20292701 --seed-stride 32 --execution-order cpu-first \
  --cpu-backend dictionary --hybrid --candidate-limit 4 --hybrid-rounds 0 \
  --gpu-stages maxmin --pipeline-cpu-stages \
  --heterogeneous-candidates --heterogeneous-replay --cpu-basis-proposals \
  --compact-capture --candidate-ranking count --restricted-bucket \
  --max-repair-batch 4 --repair-cache-size 4 --tie-policy original3
```

`--hybrid-rounds 0`なのでGPU補正は実行しない。`--repair-operators`は現CLIが必要とする対応artifact指定であり、ここで補正実行の効果を測っているわけではない。
