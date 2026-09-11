# 行列分解・三角求解の最適化 — 2026-09-07

## 対象と変更しない条件

前回のGPU数値更新を維持し、残る反復線形代数を対象にした。
保存済みmaxmin LPの32環境・step2～8を再生する開発検証である。
物理状態をGPU出力から逐次更新する閉ループdFBA、aggregate/exchange、PPO全体の速度ではない。

元LPのprimal residual ≤ 1e-5、dual violation ≤ 1e-7、relative KKT gap ≤ 1e-7と
独立direct dual gap検査を維持する。CPU参照解はGPU入力に使わず、GPU経路にCPU LP呼出しはない。
CPU比較はHiGHS 1.15.1、16 workers、各worker 1 solver thread、model/basis再利用あり。
環境を複製した見かけ上のバッチ増大は行っていない。

## 実装した3案

### 1. 前処理factorの低頻度更新（既定採用せず）

`factor_reuse_interval`を追加した。各solveの初回は現在値で分解し、同じsolve内の後続反復で
旧factorとmatching scalingを前処理として使う。現在のratioを含むNewton作用素を変更せず、
実際の全Newton残差がforcing基準を満たさなければ、同じ状態で現在のGPU factorを作り直す。
その後の方向受理・実際のmerit減少・元LP認証も省略しない。既定interval=1は従来動作。

4環境では精度を維持したが、求解時間は対照0.282/0.234秒に対して1.036/0.857秒へ悪化した。
factorは10/8回のままで、候補補正・再分解・追加反復が増えたため、32環境試験へ拡大せず採用を見送った。
準Newton的な低頻度更新が常に速いとはせず、本構成での負の結果として保存した。
[着想の参考: Gondzio・Sobral](https://www.maths.ed.ac.uk/~gondzio/reports/qnIPM.pdf)

### 2. 有限値検査・入出力変換の融合（実験オプション）

`src/gpu_solve_guards.py`に2個のCUDAカーネルを実装した。入力のlane全体を検査し、不正laneを
cuDSSに渡す前にゼロ化する。求解後は不正入力または非有限出力のlane全体をNaNにし、
後段の数値・LP認証で受理できないようにする。ゼロ化した値を正解扱いする変更ではない。
不正入力・出力の回数はGPU上に記録し、返す配列は次の求解で上書きされない独立配列である。

非連続・負stride・broadcast入力、複数RHS、NaN/±Inf、native buffer変更拒否をテストした。
32環境の初回試験はhot求解0.642/0.485秒、更新等込み0.808/0.599秒で、
総時間の明確な優位を確認できなかった。最終レイアウト比較には混ぜていない。

### 3. cuDSSのnative実行レイアウト変更（単独比較）

`factor_layout='block_diagonal'`を追加した。論理的には32個の独立LPのまま、
cuDSSには非対角ブロックが0の1個の大きな疎行列を渡す。目的関数・交換フラックス・
環境間の資源共有を変えるものではない。

論理的なshape・KKTマッピング・認証bufferは従来通り。nativeのCSR座標だけを初回に作り、
同じ連続数値bufferで各環境の現在の係数を更新する。native CSRも独立に組み立てた座標と
初期照合し、GPU数値更新のpreflight検査に含めた。インデックス破損時は実行前に拒否する。
現段階はnrhs=1のみ対応、int32容量を超える形状は構築前に拒否する。

cuDSSの公開仕様は同一構造に対するanalysis再利用を許すが、それだけで特定レイアウトの
性能向上を保証するわけではない。本環境に導入済みの0.7.x ABIで実行テストしている。
[NVIDIA cuDSS実行仕様](https://docs.nvidia.com/cuda/cudss/functions.html)

## 検証方法

`scripts/run_ipm_layout_compare.py`で、uniform → block_diagonal → CPU16の順に3回ずつ順次実行する。
GPU/CPU実験・pytestを重複実行しない。順序は固定で、ランダム化クロスオーバー試験とは呼ばない。
各試行の現在入力hash・環境順序・ソースhash・精度条件を照合してから中央値・範囲を集計する。
初回構築費と更新・求解・検証を分離表示し、初回費を隠してCPU超えとは主張しない。
信頼区間や有意差は算出せず、n=3は同一入力の実行反復数である。

全回帰は **1,555 passed / 4 skipped / 11 warnings、36.46秒**。
skipは2枚目GPU必須の検査、warningは既存のSciPyオプションと意図的CSR変更によるもの。
[回帰XML](../results/pf_ipm_linear_algebra_regression_20260907.xml)

## 実験データ

- [factor低頻度更新の対照](../results/pf_ipm_factor_lag4_control_20260907.json)
- [factor低頻度更新interval2](../results/pf_ipm_factor_lag4_interval2_20260907.json)
- [融合ガード32環境](../results/pf_ipm_fused_guards32_234_20260907_r1.json)
- [block diagonal初回32環境試験](../results/pf_ipm_block_factor32_234_20260907_r1.json)

## 3反復の比較結果

全方式で672/672問題が元LP認証を通過した（32環境×7時刻×3実行）。
元入力・環境順序・ソースhash・認証条件は全試行一致した。以下は壁時計時間の中央値。
各行は32個のLPのバッチ処理であり、1件のLP、PPOの1 timestep、120step rolloutの時間ではない。

| 保存時刻 | GPU uniform 更新・求解・検証等 (s) | GPU block diagonal 同範囲 (s) | CPU16 更新・求解・検証 (s) | GPU内比較の短縮率 |
|---|---:|---:|---:|---:|
| 2（初回構築込み） | 21.322 | 21.570 | 0.752 | −1.2% |
| 3 | 0.817 | 0.725 | 0.512 | 11.2% |
| 4 | 0.618 | 0.558 | 0.382 | 9.8% |
| 5 | 0.613 | 0.556 | 0.315 | 9.3% |
| 6 | 0.678 | 0.614 | 0.377 | 9.4% |
| 7 | 0.563 | 0.515 | 0.328 | 8.5% |
| 8 | 0.620 | 0.558 | 0.359 | 10.0% |

後続6時刻の合計（実行ごとの合計の中央値）はuniform 3.928秒、block diagonal 3.523秒、
CPU16 2.249秒。新GPU方式は従来GPU方式から10.3%短縮したが、CPU16に対しては約1.57倍の時間を要した。
3実行の範囲はそれぞれ3.872～3.931秒、3.507～3.611秒、2.247～2.307秒。
初回構築込みsequence lifecycleの中央値は25.818/25.748/3.074秒で、全体の短縮はまだ小さい。
ファイル入力読込は全方式で別計時である。

**採否:** block diagonalを32環境の繰返しmaxmin検証で使える改善候補として残す。
low-frequency factor案と融合guard案は改善を確認できなかったので重ねず、全体の既定値は変更しない。
PPO/全stageの未検証部分へ適用して速度優位を主張しない。CPU超え・完全GPU完結は引き続き未達。

再現コマンド（出力名は未使用のものを指定）:

```sh
CUPY_ACCELERATORS= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-cuopt-26.8/bin/python scripts/benchmark_ipm_sequence.py \
  --mode gpu --batch 32 --steps 2 3 4 5 6 7 8 \
  --reuse-numeric-workspace --device-numeric-updates --restart-mu 0.00001 \
  --factor-layout block_diagonal --output results/new_block_comparison.json
```

[3反復の集計（各rawファイルとSHA256を含む）](../results/pf_ipm_layout_compare_summary_20260907.json)

## 次の課題

factor更新間隔を延ばすだけでは現在Newton系からのずれが大きく、三角求解数が増えることを確認した。
次は、同じ初期状態からのNewton方向について、旧factor補正の品質と費用を事前に測り、
安価な補正で残差基準へ到達する構造が本当にあるかを評価する。単純な再利用間隔の探索は優先しない。
同時に、初回構造検証・native分析の費用を分離して短縮する必要がある。

CUDA Graph化は今後の候補だが、native factorの内部allocation・寿命を検証せずcaptureを再利用しない。
公式仕様でもanalysisは同期処理であり、既定のcudaMalloc/cudaFreeにはcapture上の制約がある。
allocatorを含めた別の実装・検証が必要で、今回の実装完了項目には含めない。
[NVIDIAのGraph・メモリ寿命に関する仕様](https://docs.nvidia.com/cuda/cudss/general.html)
