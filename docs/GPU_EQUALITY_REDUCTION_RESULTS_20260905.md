# 等式保存型縮約とGRU–GPU補正：実装・検証記録

## 結論と検証範囲

固定の二項同次等式から独立な2,411本を消去し、GPU上で主解・双対解を元LPへ復元して認証する経路を実装した。実LPの代数的同値性検証は36/36合格。一方、縮約したGRU＋PDHGによる独立4環境×41ステップのexchange replayは0/164合格であり、**精度を維持したCPU超え・環境全体の完全GPU化は未達**である。未認証解でPPOや培養軌跡を進めてはいない。

本検証の41ステップは、保存されたCPU生成状態・上流2LP結果から作られたexchange入力を順に処理するもの。各方式は自分の認証済み解だけを履歴に使用する。現在/過去のCPU正解はGPU初期値に渡していないが、上流も含めてGPUで状態生成したclosed-loopではない。120ステップの24時間培養やPHA終点誤差の達成を意味しない。

## 構造監査

教師16環境×60ステップ×3段階の2,880 LPと、独立診断4環境×step1/2/41×3段階の36 LPを監査した。末尾の未使用補助列を除けば、等式行列は全2,916件で同一hash。等式4,651行のRHSは全て0である。

1非零行336、2非零行2,444。この2非零行のうち独立forestは2,411本、残り33本は閉路等であり、全2,444本を独立と数えていない。v1実装は2,411本だけを消去し、単項等式・閉路・過大な重み比の成分は保守的に残す。全等式の零空間を得たわけではない。

training全体で共通のゼロ固定bounds列は11本。一部条件では13本となるが、切替列5366/5381を全期間でゼロ固定してはならない。本実装は可変boundsを各LPから取り直す。生物反応・GEM・遺伝子ID・目的順序・培地条件は変更していない。

| 段階 | 元の行×変数 | 縮約後の行×変数 | 変数削減率 |
|---|---:|---:|---:|
| maxmin | 5,051×6,734 | 2,640×4,323 | 35.8% |
| aggregate | 5,051×6,734 | 2,640×4,323 | 35.8% |
| exchange | 6,330×7,373 | 3,919×4,962 | 32.7% |

代表exchangeの非零係数は31,501→24,397（22.6%減）。復元写像Tの非零数は7,373、双対復元写像は4,806。構造計算は36 LP検証内で段階当たり約0.056–0.099秒の初回host費用で、オンラインCPU LPではない。縮約率や非零係数削減率を、そのまま実行時間短縮率へ換算しない。

## 実装の要点

`E x = 0` の消去対象に対して疎行列Tを構成し、`x = T z`、`E T = 0` とする。残る行列G、目的c、上下限は毎回 `G T`、`Tᵀ c` と群ごとの上下限の共通部分へ変換する。Tの各成分は最大絶対値1に正規化し、例えば `x1690 − 5e−5 x2574 = 0` は `[5e−5, 1]` として係数の逆数増幅を避ける。

双対解は単にゼロ埋めしない。残存行の双対yについて `q = c − Gᵀ y`、縮約reduced cost `r' = Tᵀ q` を求め、縮約boundsの端点を決めた元の変数へnormalを割り当てる。負のT重みによる上下限の反転も考慮する。その後、事前構築した疎なtree写像で消去等式の双対を復元する。元LPで残差・双対符号・相補性を再検査するため、縮約側だけで合格した候補を採用しない。

主残差≤1e−5、双対違反≤1e−7、相対KKT gap≤1e−7は変更していない。GPU反復・主双対復元・元LP残差演算にCPU optimizerは入っていない。固定写像の構築、動的LP変換、GRU特徴作成、停止判断、返却配列にはhost処理が残る。

関連実装：`src/lp_equality_reduction.py`、`src/gpu_reduced_pdhg.py`、`src/temporal_gpu_lp.py`。縮約は `equality_reduction=True` / `--equality-reduction` の明示opt-inで、従来経路やPPO既定は切り替えていない。

GRUの予測とhidden更新を`propose`/`commit`へ分離した。GPU補正例外時にはhiddenを進めず同一stepを再試行できる。正常に返った未認証候補は解キャッシュへ入れないが、CPU生成入力を順送りする本replayでは観測入力の履歴だけは進める。generatorでのresetがhiddenへ伝わらない不具合も修正した。非ニューラル対照の不要なlatent行列演算も除去した（下記41step結果のsource snapshotはこの小修正前）。

## 実測：CPU同値性とGPU性能を区別

36 LP検証では、(a)参照主双対を縮約・復元して36/36、(b)縮約LPを別途CPU HiGHSで解いて復元して36/36、いずれも元LP証明が合格した。多重最適性によりfluxやdual座標が異なっても、元LPの制約と最適性で判定している。この検証にCPU参照を用いることと、オンラインGPU推論への正解混入を混同しない。

RTX 4060 Laptop、exchange、4環境×41ステップ、各LP最大2,048 PDHG反復、256反復ごと認証、FP64、omega=1。CPU対照は4 worker・各LP 1 threadのpersistent HiGHSで、辞書初期化を使うさらに強い既存CPU対照ではない。時間は各batch APIの合計で、特徴・変換・転送・推論・反復・認証・返却を含み、trace読込とモデル読込は含まない。

| 方式 | 164 LPの処理時間 (s) | 元LP合格 |
|---|---:|---:|
| CPU persistent HiGHS | 10.166 | 164/164 |
| 等式縮約＋cold GPU PDHG | 29.926 | 0/164 |
| 等式縮約＋既存GRU＋GPU PDHG | 29.964 | 0/164 |

これは1回の開発測定で、CIや一般的な速度優位は主張しない。GPU時間は**失敗候補の処理時間**であり、同等品質の計算完了時間や高速化倍率ではない。新しい縮約座標へGRUを再学習した実験ではなく、既存rank32モデルの出力を縮約・補正する接続試験である。

step41の最大主残差はcold0.148/GRU1.870、双対違反は0.00487/0.0315、相対KKT gapは241.97/121.05。依然として認証基準から遠い。

GRU step41のbatch 0.7233秒内訳は補正反復0.6673秒（約92%）、元LP認証0.0212秒、構築0.0222秒、特徴等host準備0.00680秒、GRU推論・復号0.00247秒。反復時間にはPythonからのkernel起動も含まれるので、これをGPU演算器の占有時間と呼ばない。ネットワーク推論のみを速くしても主要費用は除けない。

## データと再現

独立cold診断8 LP（step1/41）で、絶対相補性gapの99.5%以上は変数bounds側だった。上位8列だけではその約9–17%にとどまるため、少数の異常列だけではなく広域のactive set同定が不足している。box自体の主実行可能性違反は0、主残差は未消去の多項等式に残る。代表はstep1の行960（1,193非零、最大係数4,800）、step41の行2307等である。縮約と元LPの残差差は主残差約2.3e−15以下、gap約2.0e−10以下で、復元の失敗が未合格の主因ではない。

- `results/pf_lp_equality_audit_20260905.json`：構造監査、入力とsourceのhash。
- `results/pf_equality_reduction_cpu36_20260905.json`：3段階・step1/2/41の縮約/復元証明。`.sources/`に実行source保存。
- `results/pf_reduced_pdhg_cold_mean4x3_20260905.json`：初回短期pilot。cold/meanとも0/12。
- `results/pf_reduced_pdhg_cold_gru4x41_20260905.json`：因果的exchange replay、初期化方式別の全step診断。`.sources/`に実行source保存。
- `results/pf_reduced_pdhg_residuals_step1_41_20260905.json`：元LP/縮約LPの残差内訳、上位行・列、誤差と時間の診断。

```bash
CUPY_ACCELERATORS= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-cuopt-26.8/bin/python scripts/benchmark_temporal_lp.py \
  --artifact results/pf_gru_exchange_rank32_scaled_20260905 \
  --trace results/pf_lp_trace_dev4x60_20260905 \
  --output results/NEW_unique_replay_name.json \
  --steps 41 --environments 4 --iterations 2048 --check-interval 256 \
  --modes cold gru --equality-reduction
```

## 固定反復の起動負荷：CUDA Graphの限定試験

CuPy 14.2.0のcuSPARSE wrapperはstream capture内の呼出しを拒否した。この失敗を保存し、環境やライブラリ内部を変更せず、1 warp/CSR行のFP64疎行列積カーネルを別の明示方式として実装した。同じカーネル・固定buffer・転置行列で、Pythonから逐次起動する場合とCUDA Graphで64反復ずつ起動する場合を比較した。

固定されたexchange入力4 LP、計2,048反復、交互順の3 paired trialで、平均の反復処理時間は **0.31696→0.17630秒、平均時間比1.798倍**。Graph固有のcapture/upload/初回launch費用は0.01409秒。共通初期化・warmupも加えた32chunk分の算術見積りは0.94067→0.81408秒で約1.16倍となる。ただし後者は実測した各費用の合算で、連続したdFBA計算の実測時間ではない。反復時間には元LP認証や入力更新を含めていない。

同じ2,048反復の既存cuSPARSE GPU補正器との比較では、元空間x/yの最大差は9.24e−14/1.00e−11、元LPの残差と合否も一致した。これはGPU演算実装同士の比較で、CPU HiGHS正解への精度を示すものではない。**元LP証明は両者とも0/4合格**であり、1.80倍は固定された反復処理の速度比に限る。GPU使用率・有効FBA throughput・CPU超えの実証値へ読み替えない。

成功記録は `results/pf_pdhg_graph_csr4x64_2048_v2_20260905.json`。初回cuSPARSE capture非対応は `pf_pdhg_graph_chunk4x64_20260905.json`、診断キーの不一致で停止した初回CSR試験は `pf_pdhg_graph_csr4x64_2048_20260905.json` として保存し、成功データに混ぜていない。実装は `scripts/benchmark_pdhg_graph_chunk.py` の `--spmv-backend csr-kernel --chunk 64 --launches 32 --repeats 3`。現時点では診断スクリプト内の固定入力試作で、GRU service・PPO・動的入力更新には統合していない。

## 回帰試験

従来cuOpt/CuPy経路を710件、新時系列・縮約・監査を別processで120件、残差分解をCPU-onlyで5件実行して全件合格した。GPU縮約6件は最初の2群で重複するため、独立テスト数は829件。CPU optimizer呼出し禁止、負のT重み、環境別bounds、元LPに戻すと不合格になる反例、stale等式/RHS、GRU例外後再試行、generator reset、非ニューラル対照の不要な行列演算禁止を含む。

旧conditional graph/cuOptと新しいTorch/CuPyを同一processで混在させる依存互換性問題は、この実装では解決したとみなさない。従来と新規のテストprocess分離を維持し、インストール済みライブラリの更新・置換は行っていない。

## 文献との対応

[DC3（Donti et al., ICLR 2021）](https://arxiv.org/abs/2104.12225)は部分予測を等式適合解へ補完し、不等式の補正も学習へ含める構成を示す。本実装はその設計に沿った限定的な等式保存層であり、DC3の全面再現や本dFBAでの速度実証ではない。

変数消去と主双対postsolveは既存の最適化前処理に属する。[GALAHAD公式PRESOLVE文書](https://ralna.github.io/galahad_docs/html/Python/presolve.html)も、問題縮約後の解を元の形式へ復元し、主双対・相補性を扱う必要を説明する。本コードの同次forestとGPUバッチ復元の組合せについて、独自の新規性が証明されたとは主張しない。
