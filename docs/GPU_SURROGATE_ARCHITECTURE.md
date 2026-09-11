# GPU FBA surrogate architecture

> **2026-09-11時点の注記**: 本書のベンチマークは *L. plantarum* (WCFS1) を第3菌種と
> していた当時の3 GEM構成で測定されたものです。現行の第3菌種は
> *P. freudenreichii* (Pf) に変更されており(詳細は
> [PROJECT_MANAGEMENT.md](PROJECT_MANAGEMENT.md) Phase D参照)、GPUサロゲートの
> アーキテクチャ・手法自体は菌種に依存せず有効ですが、下記のL. plantarum固有の
> 数値(反応数・実行時間等)はPfモデルで再測定されていません。

## 結論

このプロジェクトの3 GEM（OR16、NS21、*L. plantarum*）は、1件ずつの
厳密LPをcuOptへ送る構成ではRTX 4060上でCPU HiGHSより遅かった。そのため、
RL rolloutでは「厳密FBA解の有限辞書をGPU上で一括評価する」方式を採用し、
最終評価・論文用数値はCPU HiGHSで再計算する。

```text
SubprocVecEnv workers
  └─ LP parameters [lb, ub, objective, sense]
       └─ single GPU owner / micro-batcher
            ├─ dictionary fluxesを全lb/ubで並列filter
            ├─ feasible候補だけobjectiveを並列評価
            ├─ best feasible fluxを返す
            └─ candidateなし / OOD / residual異常 → CPU HiGHS
```

辞書の各列はHiGHSで得た完全な反応フラックスである。したがって選択候補は
`S v = 0` を保持する。実行時にも全反応境界、相対物質収支残差、有限値を検査する。
候補を混合するニューラル方式も比較したが、動的境界を越えるため安全採用率が低く、
GPU辞書最適化を既定とした。

## 先行研究との対応

- FBAをANNサロゲートへ置換し、代謝切替を含む動的シミュレーションを高速・安定化
  できることが報告されている。
  [Coupling FBA with reactive transport modeling through machine learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC11840022/)
- dAMNはニューラル成分をdFBAへ組み込みながらGEMの化学量論制約を保持する。
  [dAMN: a genome-scale neural–mechanistic hybrid model](https://pmc.ncbi.nlm.nih.gov/articles/PMC13184965/)
- 保存則を固定化学量論層として構造へ埋め込む考え方は、化学系の
  physics-constrained surrogateで検証されている。
  [Conservation laws in a neural network architecture](https://gmd.copernicus.org/articles/15/3417/2022/)
- NVIDIAはPDLPを非常に大規模なLP向けとし、小〜中規模または高品質基底解には
  dual simplexを案内している。LP BatchSolveは廃止予定で、独自並列化を推奨する。
  このためSubprocごとのcuOpt呼出しではなく、単一GPU所有サービスを実装した。
  [cuOpt LP/QP features](https://docs.nvidia.com/cuopt/user-guide/latest/lp-qp-features.html),
  [cuOpt convex features](https://docs.nvidia.com/cuopt/user-guide/latest/convex-features.html)

## RTX 4060での受入試験

128環境状態をHiGHSで生成し、102解を辞書、26解をholdoutに使用した小規模試験。
これは本番精度の保証データセットではなく、実装経路の受入試験である。

| GEM | CPU HiGHS (LP/s) | GPU安全採用率 | objective MAE | 最大相対 `S v` 残差 |
|---|---:|---:|---:|---:|
| OR16 | 6.85 | 91.4% | 0.00433 | 4.9e-8 |
| NS21 | 6.74 | 96.9% | 0.00388 | 5.5e-8 |
| *L. plantarum* | 26.28 | 98.4% | 0.000166 | 9.8e-8 |

24ステップのwarm-start済み単一環境rolloutでは、HiGHS 12.49秒に対しGPU
サロゲート2.23秒（5.61倍）。72 FBA中65件をGPUで採用し、7件はHiGHSへ戻した。
辞書外の状態を追加して採用率を上げることが次の主要改善点である。

結果は `results/fba_surrogate_gpu_lp_rtx4060.json` と同名PNGに保存する。

## 実行方法

```bash
# 厳密ラベル生成とGPU辞書作成
python3 scripts/train_fba_surrogate.py \
  --sbml-dir models/sbml \
  --output-dir models/fba_surrogate_gpu_lp \
  --samples 2048 \
  --episode-steps 256 \
  --decoder-mode feasible_dictionary_optimizer \
  --device cuda

# CPU/GPU比較グラフ
python3 scripts/benchmark_fba_surrogate.py \
  --artifact-dir models/fba_surrogate_gpu_lp \
  --device cuda \
  --output results/fba_surrogate_rtx4060.json

# 複数環境RL
python3 main.py train \
  --sbml-dir models/sbml \
  --solver-backend surrogate \
  --surrogate-dir models/fba_surrogate_gpu_lp \
  --surrogate-device cuda \
  --n-envs 16 \
  --device cuda
```

`n_envs > 1`ではGPUごとに1つの推論サービスを起動する。各環境は既定2 msだけ待ち、
同じ菌種の要求を1バッチへまとめる。CUDA contextと辞書はGPUごとに1コピーとなる。
`--gpu-ids 0,1,2`を指定した場合はGPUごとにサービスを1つ起動し、環境workerを
round-robinで分配する。

## 科学的な運用制約

1. サロゲートはRL探索専用。最終スコア、図、統計検定は `--solver-backend highs`。
2. holdout軌道は訓練辞書と別seed・別方策で作り、辞書内誤差だけを報告しない。
3. 本番学習では定期HiGHS監査を有効にする。目的値監査失敗は厳密解へ置換される。
4. acceptance、OOD、bound violation、audit failureを結果JSONで監視する。
5. 辞書増加時はVRAMと `batch × candidates × reactions` の一時tensorを測定する。
   3枚構成では環境をGPU単位に分け、各GPU内でmicro-batchする。
