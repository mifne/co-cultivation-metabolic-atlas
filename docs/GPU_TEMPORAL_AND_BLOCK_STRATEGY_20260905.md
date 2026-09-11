# 時系列学習とGPU集約LP：実装・検証状況

## 結論と対象

現時点でRNN/LSTM/GRUを使った時系列LP予測は未導入である。`src/gpu_neural_basis_proposal.py` の `NeuralBasisProposal` は現在LPの特徴量を入力する全結合tanhネットであり、履歴を記憶する隠れ状態を持たない。最新の推奨ベンチマーク経路はcompact GPU辞書の候補選択・元LP証明とpersistent HiGHSの組み合わせで、ニューラル候補選択の試作が存在することと、今回の計測でそれを使用していることも区別する。

今回の高速化対象はOR16・NS21・P. freudenreichiiの元の3段階共同LPを使うdFBA環境計算である。PPOの方策モデルをRNN化することと、環境内FBAをRNNで初期化することは別の変更である。方策をRNN化してもLP呼出しが自動的に減るわけではない。

## RNNを加える場合の設計

現在の完全なLP入力 `(A, b, lower, upper, c)` が与えられれば、そのLPの最適解集合は現在入力で定まる。履歴はLPに新しい生物学的記憶を加えるためでなく、連続する問題の初期解・active set・補正費用の予測に使う。培養モデルに酵素発現遅延等の実測された記憶を加える研究は、数値加速とは別途検証が必要である。

候補構成は以下とする。これは設計案であり、学習済みモデルや速度結果ではない。

1. 各環境の現在状態・操作・前回からの変化量・前回の認証済み低次元解・LP補正履歴を入力する。環境IDごとにGRUの隠れ状態を分離し、episode resetで消去する。環境バッチは並列化し、時刻間依存は維持する。
2. 既存辞書の番号を選ぶだけでなく、低次元解の補正量または追加すべきactive-set候補を予測する。全候補が無効な状態では、候補番号の分類器だけを改善しても有効解は作れない。
3. 固定された化学量論構造を使って必要なフラックス空間へ展開し、元LPの主実行可能性・双対・相補性を検証する。誤差の大きいものだけGPU補正または測定済み費用に応じたCPU再最適化へ進める。RNNの信頼度だけでは解を採用しない。
4. 学習目的は単なるCPUフラックスとのMSEだけでなく、固定回数の補正後の残差・目的値・必要な補正量を評価する。多重最適解では異なるフラックスでもLP最適であり得る一方、異なる交換フラックスが長期の培養結果を変えるので、終点検証も維持する。

比較群は「現在入力MLP」「単純な前解再利用」「GRU」とし、同じ候補空間と補正器を使う。trajectory seed単位で学習・開発・最終評価を分離する。時系列の隣接行をランダム分割しただけの評価や、現在のCPU正解をGPU初期化へ入れた比較は行わない。評価指標は推論時間だけでなく、補正込みオンライン総時間、CPU復帰数、全LP証明、120ステップのPHA・菌体量・PHV終点差とする。教師データ生成・学習は別計上する。

関連文献は[LSTMで最適化アルゴリズムを学習する研究](https://arxiv.org/abs/1606.04474)、[ニューラルwarm startとfixed-point反復を結合する研究](https://www.jmlr.org/beta/papers/v25/23-1174.html)である。これらの成果を本dFBAの速度倍率へ転用しない。[FaureらのAMN](https://pmc.ncbi.nlm.nih.gov/articles/PMC10400647/)はニューラル・機構モデル結合の根拠だが、本件のRNN時系列LP加速を実証した論文ではない。

## 今回追加した再現用LPトレース

`scripts/capture_dfba_lp_trace.py` はCPUの認証済み解だけで培養状態を進め、各時刻の元LPを保存する。`src/lp_trace.py` はCSR・RHS・上下限・目的係数の読み戻しhashを検証する。参照主双対解は採点専用で、GPU初期値として使わない。

`results/pf_lp_trace_dev4x60_20260905/manifest.json` はseed20293401–04、4環境×60ステップ、720 LPを完了し、全入力の保存・読戻し一致を確認した。これは開発診断専用で学習辞書に加えていない。I/O・圧縮を含むcaptureは77.426秒、環境構築は10.882秒だった。これらをCPU/GPU速度比較に用いない。LPトレースは環境の全内部状態checkpointではない。

## GPU block-diagonal試作

`src/gpu_block_lp.py` は同時刻・同段階の独立環境LPをblock diagonalに結合し、目的関数の和をcuOptの単一GPU solveで解く。環境間に新しい栄養共有制約は作らない。host配列組立・cuOpt presolve・返却APIを含むため完全device residentとは呼ばない。Concurrent、CPU simplex、crossoverは使用しない。

同一Aを共有する[BatchLP](https://arxiv.org/html/2601.21990)と異なり、本モデルでは環境間でbiomass依存の行列係数も変わる。したがって各 `A_i` をそのまま保持する。GPU化の候補は計算を増やすことではなく、小さなLPごとの起動費用を大きな疎LPへまとめることである。[cuOpt公式資料](https://docs.nvidia.com/cuopt/user-guide/latest/lp-qp-features.html)のsolver特性を参照した。

各blockについて元LPの主残差≤1e-5、双対違反≤1e-7、相対KKT相補性gap≤1e-7をGPUで検証する。全体のoptimal statusや平均gapだけでは合格にしない。試作の既定ではglobal optimalも要求し、時間上限終了の部分採用はまだ実装していない。有限な途中解が独立block証明を通る場合の部分採用は、今後明示的な別規則として検証できる。

### 4環境の初期診断

同時刻4環境の保存LPをcold startで比較した。CPUは4 workerのHiGHS、GPUはFP64 PDLP Stable3/presolve2/crossover off。各段階3秒のGPU診断budget。CPUの既存基底を使う長期比較ではなく、環境更新も含まない局所診断である。

| step | 段階 | CPU秒 | GPU秒 | 元LP証明を含むGPU採用 |
|---:|---|---:|---:|---:|
| 1 | maxmin | 0.159 | 1.221 | 0/4 |
| 1 | aggregate | 0.249 | 3.030 | 0/4 |
| 1 | exchange | 0.365 | 3.035 | 0/4 |
| 41 | maxmin | 0.193 | 0.900 | 0/4 |
| 41 | aggregate | 0.329 | 3.030 | 0/4 |
| 41 | exchange | 0.491 | 3.033 | 0/4 |

`results/pf_gpu_block_pdlp_dev4_20260905.json`。maxminはcuOpt内部でoptimalだったが、外部相補性gap約7e-4～1.8e-3が基準を超えた。後段は時間上限。Barrierも同じ6ケースで時間上限、採用0/4だった（`pf_gpu_block_barrier_dev4_20260905.json`）。不合格時間を速度倍率にしない。初期診断版GPUは既に正規化した配列を受けたため、CPU側の再正規化との費用差が残る。後続スクリプトは両側に同じrequest形式を渡し、GPU側の正規化・結果copyも総時間へ含めるよう修正した。

### 上限到達の別の双対証明

maxminでは目的は `min -z`、`upper(z)=0.005` である。実行可能な解がこの上限に到達すれば、行の双対変数をすべて0とし、reduced costを元の目的係数cとすることでも最適性を証明できる。これは返却双対の丸め・違反の切捨てではない。元LPに対して別の正当な双対解を構成するものである。上限未到達・実行不可能・必要な有限境界の欠落は同じ外部基準で棄却する。

このopt-in証明を加え、内部停止を1e-7とした4環境maxminではstep1/41とも4/4合格した。しかしGPU1.063/0.796秒、cold CPU0.175/0.188秒で、速度優位はない（`pf_gpu_block_pdlp_box_dev4_20260905.json`）。証明追加と内部停止設定を同時に変えたため、速度差を証明処理単独の効果とは呼ばない。これも前述の正規化費用修正前の診断である。

### 32環境での追試

`pf_lp_trace_dev32x41_20260905/manifest.json` に新規seed20293601–32の32環境×41ステップ、3,936 LPを保存した。途中状態への不正な飛越しはせず、CPUで前段から状態を進めた開発診断である。I/O込みcapture338.685秒は速度比較に使わない。

`pf_gpu_block_pdlp_box_dev32_20260905.json` は両側で元requestを正規化し、GPUの入力組立・結果copy・元LP証明まで計時した。GPU内部停止1e-7、各solve上限3秒である。CPUは各行で新規backend、GPUライブラリ初期化は別計上で再利用するため、この表はcold-model局所診断でありpersistent CPUとの厳密な速度倍率ではない。

| step | 段階 | CPU秒 | GPU秒 | 元LP証明を含むGPU採用 |
|---:|---|---:|---:|---:|
| 1 | maxmin | 1.333 | 2.686 | 32/32 |
| 1 | aggregate | 2.200 | 3.208 | 0/32 |
| 1 | exchange | 3.309 | 3.264 | 0/32 |
| 41 | maxmin | 1.670 | 3.051 | 32/32 |
| 41 | aggregate | 2.408 | 3.204 | 0/32 |
| 41 | exchange | 3.423 | 3.209 | 0/32 |

後段は時間上限で不合格なので、CPUより短い行があっても高速化とは扱わない。maxminは上限到達の解析的双対証明によって全件採用できたが、32環境でもCPUより遅かった。

さらに目的変数をその有限最適側上限に固定し、目的係数を0として実行可能性問題を解く候補生成を追加した（`--box-face`）。採用には必ず元の目的係数・境界値・行列で証明を取り直す。上限面で解けなかったことを元LPの実行不可能性とは扱わない。`pf_gpu_block_face_dev32_20260905.json` ではstep1/41とも32/32採用したが、GPU2.494/2.356秒、CPU1.393/1.594秒で、速度優位はない。内部停止1e-6へ変更しても、外部の元LP精度基準は緩めていない。

## 共通host最適化の切り分け

`--frozen-inputs` は固定GEMの分類・静的bounds・目的係数を一度だけ配列化し、毎ステップの状態依存boundsと目的切替だけを評価する。COBRAへの書込みとLP配列への読戻しを省くが、生物学的反応や境界式は変更しない。モデルを途中編集しない明示opt-in契約で、初回の完全な整合性検査は環境構築費用として別計上する。CPUにも同じ変更を適用した。

同じsource/model/bank、4 worker、seed・操作列・実行順を揃え、4環境×8ステップの2組を比較した。各環境1.6時間の仮想培養であり、PPO学習全体の所要時間ではない。

| 組 | CPU従来 → 配列入力（秒） | Hybrid従来 → 配列入力（秒） | Hybrid時間短縮 |
|---|---:|---:|---:|
| 0：seed20293801–04、CPU先行 | 4.993 → 4.148 | 5.344 → 4.622 | 13.51% |
| 1：seed20293805–08、GPU先行 | 5.774 → 4.120 | 5.105 → 4.201 | 17.72% |

変更前後でPHA・菌体量・PHV比率・全培地終点が完全一致し、CPU LP数とsimplex反復数も一致した。元LP証明も全件合格した。派生結果は `pf_array_only_pair_4x8_20260905.json`、生データは `pf_array_support_off_4x8_20260905.json` と `pf_array_only_4x8_20260905.json`。2組の開発測定で、設定間は従来の後に配列版を測ったため順序・温度等の交絡が残り、信頼区間や一般的な高速化率は主張しない。配列版同士ではCPU4.148/4.120秒に対してHybrid4.622/4.201秒で、GPU追加による速度優位は未確認である。

CPUのexchange性能制約行の疎構造変更を `changeCoeff` だけで扱うオプションも追加した。ただし配列化との併用ではexchange再構築8→4件/組に減っても、HybridのCPU反復は13,110→23,686、11,477→21,557へ増え、総時間5.344→5.443秒、5.105→5.294秒と悪化した。単に再構築回数を減らすのでは不十分であり、このオプションは既定offを維持する。集計のdirect/grouped診断の二重計上を修正した `pf_array_support_pair_4x8_v2_20260905.json` を正とし、旧派生集計のCPU要求数・反復総数は使用しない。生データ自体への修正はない。

ステージ別に調べると、exchangeの反復増分20,696回のうち20,605回（99.6%）が、両組を合わせた8件の支持切替時に発生した。古い基底を維持した切替行は1,557–3,393反復、従来の再構築＋辞書初期化では26–291反復だった。aggregate反復は不変であり、単に全計算が不安定化したわけではない。これは「直前の解ほど良い初期値」とは限らない例で、時系列学習では相転換の検知と補正費用の評価が必要になる。

続く `frozen_community_inputs_4x60_20260905.json` では新規seed20294001–04、4環境×60ステップ（各環境12時間）の全720 LP入力hashと全終点が完全一致した。CPUだけの同値性診断であり、cProfileと入力hash計算を含む時間はGPU比較には用いない。中盤まで同値性は確認できたが、120ステップ・多環境での配列版の速度は未測定である。

## 残る採否判断

GPU block-diagonal方式はこの条件ではCPUを上回らず、標準経路へ統合しない。固定配列入力は共通最適化として有望だが、短期結果だけを長期へ外挿しない。中盤を含む全LP入力の一致確認を先に行い、その後に新規seed・多環境の対照比較へ進む。

RNN/GRUについては、単純な前回解の再利用が修正反復を増やす今回の観測を、学習目的の設計に反映する。正解フラックスへの距離だけでなく、補正後残差と実行費用を評価する。現時点では時系列トレースと比較設計を用意した段階で、GRUの学習・推論経路は未実装。完全GPU化、大幅なGPU速度優越、RNNによる速度向上はいずれも未達・未検証である。

## 回帰試験と実行環境の制約

関連回帰試験は最終的に688件合格（88 warnings、43.56秒）。ただし最初の混合実行は681件合格・7件失敗であり、cuOpt実GPUの3 block試験・4 lexicographic試験が共有ライブラリ読込みエラーになった。単独ではblock13件、lexicographic19件とも合格し、次の読込み順依存を特定した。

- CuPyの時系列基底試験を先に実行すると、`/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.3.14` が先にロードされる。
- cuOpt環境内の `nvidia/cusolver/lib/libcusolver.so.11` は、その旧ライブラリにない `cublasSetEnvironmentMode` を要求する。環境内の `nvidia/cublas/lib/libcublas.so.12` には同symbolが存在する。
- cuOptのloaderが元のundefined-symbolエラーを警告にして処理を続けるため、後続importでは二次的な `libcuopt.so missing` に見える。これはGPUのLP不収束とは別問題である。
- 通常のpytest collection後、cuOpt実GPUを含む `test_gpu_block_lp.py` を先に実行してから残りを実行すると688件すべて合格した。cuOptを全moduleより先に無条件importする策は、greenlet側のC++ ABI競合もあり一般的な修正としては採用しない。

これは依存関係の恒久修正ではなく、確認済みの暫定実行順である。次のGPU統合前にCUDA数値ライブラリ/C++ runtimeの互換性を揃えるか、cuOptの実GPU試験・処理を専用processへ分離する必要がある。今回ライブラリの更新・置換や、失敗テストをskipにする変更はしていない。計測済みの各benchmarkは独立processで完了しており、この失敗を含む時間は性能結果に加えていない。

再現した688件のコマンド（リポジトリ直下、WSL bash）：

```bash
CUPY_ACCELERATORS= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv-cuopt-26.8/bin/python -m pytest -q \
  tests/test_gpu_block_lp.py tests/test_frozen_community_inputs.py \
  tests/test_cpu_*.py tests/test_gpu_basis_bank.py \
  tests/test_gpu_batch_qp_targets.py tests/test_gpu_batched_compiled_backend.py \
  tests/test_gpu_[!b]*.py tests/test_community_*.py \
  tests/test_pipelined_microbatch.py tests/test_compact_benchmark_schedule.py \
  tests/test_compact_host_preparation.py tests/test_compare_pipeline_runs.py \
  tests/test_summarize_pipeline_run.py tests/test_lp_trace.py \
  tests/test_trace_script_lifecycle.py tests/test_runtime_options_pair.py \
  tests/test_uptake_bound_updates.py tests/test_hybrid_*.py \
  tests/test_selected_lp_features.py tests/test_neural_training_features.py \
  tests/test_temporal_basis_probe.py
```
