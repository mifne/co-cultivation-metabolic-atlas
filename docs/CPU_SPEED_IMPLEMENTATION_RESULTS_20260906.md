# CPU超えへ向けたGPU数値層の実装・比較結果

更新日: 2026-09-06。対象はOR16＋NS21＋P. freudenreichiiの現行LP入力。RTX 4060 Laptop GPUでの開発用数値実験であり、培養実証・PPO方策の有効性・ワークステーション上の性能予測ではない。

## 結論

厳密な等式簡約、安全策付きpredictor–corrector、初期相補性の平衡化を実装した。maxmin固定入力4環境は元LP認証4/4、GPU solve API時間1.633秒となり、旧GPU構成の5.729秒から短縮した。ただし**3.51倍は単回試験同士の暫定的なGPU内比較**であり、CPUに対する速度向上ではない。

同じ構成のaggregateは1/4、exchangeは0/4で、全3段階の認証は未達である。32独立環境のmaxmin固定入力は32/32で認証できたが、GPU 5.438秒に対してCPU 1 workerは4.135秒、4 workersは1.223秒だった。**CPU超え、全3段階の完全GPU実行、閉ループdFBA/PPOへの接続・精度確認はいずれも未達**である。

## 1. 実行構造と変更の範囲

現在の培養・PPO経路は次の依存関係を持つ。

```text
action → 流加・酸素移動・状態依存境界
       → maxmin → 成長保持下限を設定
       → aggregate → 目的保持下限を設定
       → exchange parsimony
       → 菌体・培地・PHA・pH更新 → observation / reward → 次step
```

同一環境の段階間・時刻間は独立ではない。GPUバッチ化の基本単位は独立環境の同じ段階であり、1環境の将来stepを先に並列計算する方式ではない。

今回の新GPU経路は、SHA確認済みの保存LP入力から、forest等式簡約、zero-face簡約、追加の厳密等式簡約を行い、境界dualを消去した疎Newton系をcuDSSで処理する。必要時には元Newton系を対象としたKrylov補正を行い、GPUで元座標の主双対へ復元して最初のLPを再認証する。

これは固定LPの診断経路であり、既定のPPO環境へ自動接続していない。現行環境はhost上でLPを組み立て、SciPy互換結果からCOBRA/Pandas形式のフラックスを取り出して状態を更新する。GPU数値層のCPU LP calls=0は、host構造準備・Python制御・診断・環境状態更新までGPU常駐であることを意味しない。

主な実装は [gpu_forest_ipm.py](/home/reiya/co-cultivation/src/gpu_forest_ipm.py)、[gpu_zero_face_ipm.py](/home/reiya/co-cultivation/src/gpu_zero_face_ipm.py)、[gpu_condensed_ipm.py](/home/reiya/co-cultivation/src/gpu_condensed_ipm.py)、[gpu_globalized_ipm.py](/home/reiya/co-cultivation/src/gpu_globalized_ipm.py)、[gpu_ipm_initialization.py](/home/reiya/co-cultivation/src/gpu_ipm_initialization.py)にある。

元LPの係数・反応境界・目的関数・生物モデル・報酬は変更していない。主残差1e-5、dual違反1e-7、`relative_kkt_gap` 1e-7と有限性の認証を維持した。`relative_kkt_gap`は相補性指標を`max(1, abs(objective))`で割った値であり、PHA誤差や一般的な目的値誤差の百分率ではない。内部forcing=0.1はNewton方向の相対残差条件であり、最終LPの10%誤差を認める設定ではない。

## 2. 実装とその検証結果

### 2.1 厳密な等式関係と再利用

exchangeの縮約途中の等式は1,955行で、数値QRは26行を従属候補に挙げた。保存されたbinary64係数・RHSを`Fraction.from_float`で正確な有理数として検査し、25行だけを厳密に簡約した。等式数は1,955→1,930となる。残った1行は約9.7e-17の係数差があり、近いという理由で削除していない。QR閾値だけで削除した行数は0で、完全なランク決定を主張していない。

exchangeの1環境当たりの形状は以下のとおりである。境界dualの消去はNewton線形系の代数操作であり、LPの境界を削除するものではない。

| 段階 | LP変数数 | LP行数 |
|---|---:|---:|
| 元exchange LP | 7,373 | 6,330 |
| forest簡約後 | 4,962 | 3,919 |
| zero-face等の簡約後 | 4,741 | 3,633 |
| 厳密等式25行の簡約後 | 4,741 | 3,608 |

全Newton座標17,192に対して、実際の分解次元は8,349である。maxmin/aggregateでは全Newton座標14,635、分解次元6,431となった。次元の減少率を、そのまま時間短縮率やメモリ削減率とは扱わない。

等式行列とRHSのfingerprintが一致する場合だけ、同一バッチ内で検証済み証明を再利用する。最初のexchange centered試験では、最初の証明構築2.384秒に対し、後続3環境の再利用は各0.00397–0.00419秒だった。これは構造準備部分の費用であり、物理step間の常駐更新まで実装済みという意味ではない。主双対復元・元LP再認証は省略しない。

詳細: [厳密簡約＋centered記録](/home/reiya/co-cultivation/results/pf_ipm_exact25_centered_exchange4_20260906.json)。

### 2.2 predictor–correctorとaffine予測点

同じ数値分解をaffine予測とcorrectorで共有し、各方向の元非線形残差、正値性、実際のmerit下降を確認する。失敗時は同じ状態から固定中心化方向へ1回だけ戻る。

旧affine予測は最大境界ステップを使用し、exchangeでは264 active environment-iterations中34回が中心化へ戻った。このうち27回は予測点の安全性判定によるものだった。予測用ステップだけを最大値の99.5%へ置く明示オプションを追加したところ、259 active environment-iterations中のfallbackは7回となり、同じ安全性理由による棄却は0回になった。予測点を黙ってclipする変更ではなく、最終採用ステップと元LP認証は維持している。

ただし、unsafe affineの解消だけではexchangeを認証できなかった。balanced＋PC995でもfallbackは22回残り、主な内訳は非下降方向18回、後退上限1回、affine方向精度不足3回だった。数値安全性の改善と最適性の達成は分けて評価する。

詳細: [PC](/home/reiya/co-cultivation/results/pf_ipm_exact25_pc_exchange4_20260906.json)、[PC995](/home/reiya/co-cultivation/results/pf_ipm_exact25_pc995_exchange4_20260906.json)、[balanced＋PC995](/home/reiya/co-cultivation/results/pf_ipm_exact25_balanced_pc995_exchange4_20260906.json)。

### 2.3 初期相補性の平衡化

旧初期化は`z >= 1`、`s = max(1, h-Gx)`であり、大きい無活性境界のslackが初期相補性を増大させていた。`balanced`はslackを変えず、正のdual floorだけを`1/s`へ変更する。既存dual・reduced costがfloorを上回る場合は維持するため、一般のwarm startで必ず全要素`s*z=1`になるわけではない。既定は`legacy`のままである。

exchange step1の初期記録では、slack最大値は1e6、初期μは637.6066→1.0となった。元入力1件のCPU配列集計でも、相補性の範囲は1–1e6→1–1と確認した。一方、境界項`D = Σ(z/s)`の最小値は0.002→2e-6、初期stationarityのL2ノルムは51.41→73.50となる。全残差や条件数が必ず改善する操作ではなく、正のinfeasible-start候補として比較した。

centered同士では最終相補性指標が約59–64分の1まで低下したが、認証0/4である。初期化による精度改善を、認証成功やCPU超えと読み替えない。

詳細: [balanced centered](/home/reiya/co-cultivation/results/pf_ipm_exact25_balanced_centered_exchange4_20260906.json)。

### 2.4 縮約Krylovとbarrier連動正則化

全Newton残差の判定を保ちながら、Krylovの作業座標を縮約するオプションを比較した。しかし今回のexchangeでは、centeredの9.659秒・783 solveに対し10.123秒・816 solveとなり、認証はともに0/4だった。縮約幅だけでは改善を判断できない。

balanced centeredでは、最初に失敗した各環境のμが2.35–2.48e-7まで低下し、選択δは1e-7、forcingは0.1034–0.1087で基準0.1を超えた。毎反復のbase δ=1e-6と1回retryだけでは終盤の尺度に合わない可能性を検討し、`δ = min(base, max(1e-12, 0.01 * max(active_mu)))`という明示的な実験scheduleを追加した。

このscheduleは初期から小δを固定する方式とは異なるが、実測では最終dual違反が3.54e-7–1.33e-6へ悪化し、相補性指標も4.20e-4–1.30e-3となった。5.141秒で停止したことは成功解への高速化ではない。この設定を改善済みの既定方式として採用しない。barrier連動の考え方は文献を参考にしているが、上の係数・floorは限定試験用heuristicであり、論文の正則化法そのものを再実装したとは主張しない。

詳細: [縮約Krylov](/home/reiya/co-cultivation/results/pf_ipm_exact25_centered_condensed_exchange4_20260906.json)、[barrier schedule](/home/reiya/co-cultivation/results/pf_ipm_exact25_balanced_barrier_centered_exchange4_20260906.json)。

## 3. 同じexchange固定入力のA/B比較

全行で、physical step1、4環境、同じ4つの元LPハッシュ、forest＋厳密等式25行簡約、FP64、forcing=0.1、最大240反復、Krylov上限16、retry1を使用した。既定で全座標Krylov、固定δ=1e-6とし、表に記した項目を変更した。保存されたCPU正解x/yは使用していない。各GPU条件は1回で、統計的有意差は評価していない。

| 条件 | setup秒 | solve API秒 | outer反復 | factor / solve回数 | 最終相補性指標の環境間範囲 | 元LP認証 |
|---|---:|---:|---:|---:|---:|---:|
| legacy＋centered | 4.552 | 9.659 | 118 | 134 / 783 | 5.023e-3–5.788e-3 | 0/4 |
| legacy＋PC | 4.112 | 13.103 | 71 | 92 / 1,182 | 7.169e-3–1.394e-2 | 0/4 |
| legacy＋PC995 | 4.583 | 12.877 | 71 | 86 / 1,123 | 6.843e-3–1.122e-2 | 0/4 |
| legacy＋centered＋縮約Krylov | 4.028 | 10.123 | 119 | 136 / 816 | 3.715e-3–5.775e-3 | 0/4 |
| balanced＋centered | 4.239 | 10.193 | 166 | 180 / 812 | 8.503e-5–9.554e-5 | 0/4 |
| balanced＋PC995 | 3.829 | 12.359 | 70 | 85 / 1,092 | 1.896e-5–3.626e-5 | 0/4 |
| balanced＋centered＋barrier schedule | 3.762 | 5.141 | 144 | 148 / 336 | 4.199e-4–1.297e-3 | 0/4 |

全行が未認証停止であり、表の時間からspeedupを定義しない。factor/solveはバッチ数値処理の呼出回数で、CPU LPの実行回数や成功LP件数ではない。

balanced＋PC995の主残差は最大9.900e-7、dual違反は0まで達したが、相補性は基準1e-7を満たさない。12.359秒の内訳は数値分解0.442秒、三角solve・Krylov・更新11.595秒、認証0.210秒である。反復数を減らしても、多数の内部solveが残れば短時間化には直結しない。

## 4. balanced＋PC995の全3段階

同じ構成を段階ごとの固定入力に適用した結果を示す。各段階は別の保存LPから開始しており、先行段階の新GPU解から次段階を組み立てた閉ループ実験ではない。

| 段階 | setup秒 | solve API秒 | outer反復 | 元LP認証 | 主残差最大 | dual違反最大 | 相補性指標範囲 |
|---|---:|---:|---:|---:|---:|---:|---:|
| maxmin | 3.581 | 1.633 | 33 | 4/4 | 3.000e-7 | 0 | 3.723e-8–7.548e-8 |
| aggregate | 3.453 | 12.659 | 68 | 1/4 | 1.513e-7 | 0 | 4.837e-8–4.907e-5 |
| exchange | 3.829 | 12.359 | 70 | 0/4 | 9.900e-7 | 0 | 1.896e-5–3.626e-5 |

maxminは30–33反復で認証した。解析的box-bound dualを明示的に許可し、返却する元row dualを実際に構成して再認証している。反復が生成したrow dual自体の相補性指標は0.00126–0.00668であり、これを合格と見なしたわけではない。独立host検査も4/4で合格した。

aggregateは環境0のみ68反復で認証し、主残差1.455e-10、相補性4.837e-8だった。残る3環境は方向精度不足で停止した。exchangeは全環境が同じく方向精度不足で停止した。バッチの一部成功を全段階の成功として扱わない。

初期化を変える前の厳密簡約＋PCではmaxminも0/4、6.790秒で停止していた。この記録を省かず、PC単独が一様に改善するわけではないことを残す。

データ: [maxmin](/home/reiya/co-cultivation/results/pf_ipm_exact25_balanced_pc995_maxmin4_20260906.json)、[aggregate](/home/reiya/co-cultivation/results/pf_ipm_exact25_balanced_pc995_aggregate4_20260906.json)、[exchange](/home/reiya/co-cultivation/results/pf_ipm_exact25_balanced_pc995_exchange4_20260906.json)、[初期化変更前maxmin](/home/reiya/co-cultivation/results/pf_ipm_exact25_pc_maxmin4_20260906.json)。

## 5. CPU基準と32環境比較

### 5.1 4環境の既存CPU基準

同じ元LPハッシュのCPU記録を参照する。CPUは各worker条件につき3回の中央値、GPUは各1回である。

| 固定LP | 構成 | solve時間秒 | 認証 |
|---|---|---:|---:|
| maxmin、4環境 | CPU HiGHS、1 worker | 0.5381 | 12/12 |
| maxmin、4環境 | CPU HiGHS、4 workers | 0.1522 | 12/12 |
| maxmin、4環境 | 旧GPU centered＋retry1＋解析的dual | 5.7293 | 4/4 |
| maxmin、4環境 | 新GPU balanced＋PC995＋厳密簡約＋解析的dual | 1.6332 | 4/4 |
| exchange、4環境 | CPU HiGHS、1 worker | 1.1604 | 12/12 |
| exchange、4環境 | CPU HiGHS、4 workers | 0.3440 | 12/12 |

旧GPU maxminの採用は61–64反復だった。新GPUとのsolve API比5.7293/1.6332≈3.51は、複数変更を含む暫定的なGPU内比較に限定する。旧setupは1.676秒、新setupは3.581秒であり、setup込みの比は3.51ではない。新GPUはsolve API時間だけを見てもCPUの両worker条件より遅い。

データ: [CPU maxmin4](/home/reiya/co-cultivation/results/pf_ipm_cpu_maxmin4_baseline_20260906.json)、[CPU exchange4](/home/reiya/co-cultivation/results/pf_ipm_cpu_exchange4_baseline_20260906.json)、[旧GPU maxmin4](/home/reiya/co-cultivation/results/pf_ipm_globalized_retry1_box_maxmin4_20260906.json)。

### 5.2 32独立環境のmaxmin

別の32独立環境のphysical step1について、同じ保存入力をGPUとCPUで解いた。GPUは1回、CPUは1/4 workersそれぞれ3回である。同じLPの32複製ではないが、4環境試験とは環境集合が異なるため、両表の比だけで厳密なスケーリング則を推定しない。

| 構成 | solve時間秒 | GPU setup秒 | 認証 |
|---|---:|---:|---:|
| CPU HiGHS、1 worker、3回中央値 | 4.1350 | — | 96/96 |
| CPU HiGHS、4 workers、3回中央値 | 1.2234 | — | 96/96 |
| GPU balanced＋PC995、1回 | 5.4380 | 7.9682 | 32/32 |

GPUの採用反復は31–38、主残差5.487e-8–3.756e-7、dual違反0、相補性1.050e-8–9.925e-8だった。返却主双対の独立host検査も32/32で合格した。GPU数値層のCPU LP callsは0、CPU対照は計192回の実solver実行でretryは0だった。

GPU solve APIの内訳は以下のとおりである。百分率の分母は5.4380秒で、setupは含まない。

| 区間 | 秒 | solve APIに占める割合 |
|---|---:|---:|
| 数値分解 | 0.9469 | 17.41% |
| 三角solve・Krylov・更新 | 4.2570 | 78.28% |
| 元LP認証 | 0.1366 | 2.51% |
| その他の初期化・残差・制御等 | 約0.0976 | 約1.80% |

outer反復38、数値分解38回、solve194回である。主要費用は依然として三角solve・補正・更新区間で、GPUの容量や分解サイズだけが支配的とは言えない。この区間は複数処理をまとめた時間であり、三角solveだけの費用を分離した結果ではない。

GPUはCPU 1 workerの約1.32倍、4 workersの約4.45倍の時間を要した。setup、solve、主双対転送、独立host監査、closeを含む記録済みGPU lifecycleは13.4212秒である。32環境で認証できたことはバッチ解法の適用範囲拡大だが、CPU速度優位の達成ではない。

データ: [GPU maxmin32](/home/reiya/co-cultivation/results/pf_ipm_exact25_balanced_pc995_maxmin32_20260906.json)、[CPU maxmin32](/home/reiya/co-cultivation/results/pf_ipm_cpu_maxmin32_baseline_20260906.json)。

## 6. 計測範囲と次の実装判断

- CPU時間はfresh modelの`solve_batch_wall_seconds`であり、入力検査、モデル作成、HiGHS実行、backendの元LP認証を含む。backend constructor、独立追加認証、closeは別記録。cold process・OS cache除去を主張していない。
- GPUの`ipm.total_seconds`は数値反復、postsolve、元LP認証、診断取得を含む。コンストラクター、元主双対のD2H、独立host監査は別記録。`solver_lifecycle_wall_seconds`も入力ロードやPythonプロセス起動すべてを含む指標ではない。
- CPUとGPUは時間の範囲が完全には同一でない。既存CPU記録と単回GPU記録から信頼区間や統計的有意差は主張しない。ただし今回のGPUがCPUを上回ったとする根拠もない。
- 不合格までの短い停止時間、因子分解だけの時間、同一入力を再認証するiteration0時間を「成功解へのspeedup」と呼ばない。
- 未認証のaggregate/exchangeをPPOに接続する前に、元Newtonの各残差ブロック、境界dual、残った近接依存と正則化の相互作用を切り分ける。失敗試験の反復上限を増やすだけの方針にはしない。
- 認証可能な範囲を増やした後、構造fingerprintに基づく常駐数値更新と段階別バッチアダプターを実装する。現行の反復CPU基底再利用を含む対照と比較する。
- 閉ループでは同一actionの短い軌道から確認し、状態・報酬・終了/打切り・NH4閾値等を比較する。固定LPの合格だけでは、PHA終点やPPO advantageの精度を保証しない。

文献の適用範囲と判断方針は [実装計画](/home/reiya/co-cultivation/docs/CPU_SPEED_IMPLEMENTATION_PLAN_20260906.md)を参照。PPOの精度指標と観測の変更は [PPO精度・安定化記録](/home/reiya/co-cultivation/docs/PPO_ACCURACY_SPEED_RESULTS_20260906.md)を参照する。今回の内部初期化・barrier schedule・安全策付きPCは限定的な実装選択であり、引用論文の収束保証をそのまま移したものではない。

## 7. 回帰検証の範囲

最終確認した対象テストは318件である。本体の対象suite 307件に、重複しない`test_graph_ipm_bridge.py`と`test_graph_temporal_lp.py`の11件を加えた集計であり、リポジトリ全件のpytestではない。

μの集計を`sum(comp/ng)`とし、中間総和のoverflowを避ける変更、非有限または非正のactive laneの棄却、δ変更中のfactor例外でも元設定へ戻す検証を含む。32環境試験はこの修正後のsourceを記録している。これらの回帰成功を、未実施の閉ループdFBA/PPOの精度・収束・速度の合格とは扱わない。
