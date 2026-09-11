# 3-GEM dFBA強化学習用ワークステーション導入根拠

作成日: 2026-08-28
購入案: AMD Ryzen Threadripper 9960X、TRX50、NVIDIA RTX PRO 4000 Blackwell（非SFF）×3、RAM 128GB

> **ステータス更新(2026-09-11)**: 本構成の購入は決定済み、発注済みで納品待ちです。
> 以下の本文は発注理由として作成した当時の予測・見積もりの記録であり、実機到着後の
> 実測値による再検証はまだ行われていません。

## 発注理由として使える要約

本研究では、3種の細菌GEMを同時に更新するdFBA環境を多数並列化し、PPO等の強化学習で培養条件を探索する。各環境stepでは3件のFBAを解くため、100,000 timestepsの学習だけでも少なくとも30万件のFBA相当処理が発生し、複数seed・複数条件・再現試験ではこれが反復される。

現行のIntel Core i7-12650H、RTX 4060 Laptop、RAM 48GB環境では、16並列環境においてCPU HiGHSが12.24 transition/s、GPU FBAが85.83 transition/sとなり、GPU化で7.01倍の高速化を実測した。一方、環境数を32へ増やすとGPU版でも76.15 transition/sへ低下し、16 logical threadの現行CPUが環境更新とGPUへの仕事供給を律速している。また実際の3-GEM GPUカーネルはbatch 128で平均98.3%のGPU使用率に達し、batch 64で最大5,581 FBA相当処理/sを示した。したがって、GPU計算能力を継続的に利用するには、CPU thread数・RAM容量・GPU数を同時に増強する必要がある。

Threadripper 9960Xの24 core / 48 threadを「16環境×3群」に分け、各群をRTX PRO 4000 1枚へ割り当てる。これにより、3つのseed・agent・候補生成系を独立に同時実行できる。128GB RAMは、多数のGEM worker、PPO rollout、厳密HiGHS監査、GEM構築・gap fillingを並行して行うための余裕を提供する。RTX PRO 4000は1枚24GBで、3枚合計72GBのVRAMを持つが、VRAMはカード間で共有せず、独立shardとして使用する。

## 追加したグラフ

![ワークステーション導入根拠の総合図](../results/workstation_procurement_overview.png)

- A: 16環境でCPU HiGHSからGPU FBAへ7.01倍高速化した実測と、32環境で現行CPUが飽和する実測。
- B: 実際の3-GEM FBAカーネルが平均98.3%までGPUを利用できること、および最大処理量はbatch 64であること。
- C: 現行機から購入案へのCPU thread 3.0倍、RAM 2.7倍、1枚あたりVRAM 3.0倍、総VRAM 9.0倍の容量増強。
- D: RTX PRO 4000非SFF×3の純GPU段を、現行RTX 4060 Laptop比4.08–7.24倍、計画値5.94倍とするシナリオ外挿。

![CPU・RAM増強と並列運用](../results/workstation_cpu_ram_parallelism.png)

- E: 3種GEMの厳密HiGHSラベル生成を新たに短時間測定した。96環境step＝288 LPでは、1 workerの4.67 LP/sから4 workerの7.44 LP/sまで向上し、8 workerでは起動・モデル複製コストにより6.81 LP/sへ低下した。
- F: 購入後の初期構成を、1 GPUあたり16環境、合計48環境として示した。9960Xの目的は1 runを48分割することではなく、3つの長時間探索・候補生成・厳密監査を独立jobとして同時実行することにある。

![Resource Usage](../results/resource_usage_publication.png)

**Figure 3. Hardware resource pressure in the three-GEM dFBA workflow.** (a) Mean production CUDA-kernel utilization as a function of batch size; upper whiskers denote the observed maximum and are not confidence intervals. (b) Peak VRAM and throughput measured during the same 6 s batch-saturation runs. The dashed line is the usable 8 GiB-class capacity reported for the current RTX 4060 Laptop GPU. (c) Mean system CPU utilization and maximum proportional set size (PSS) during equal-work rollouts with 16 and 32 parallel environments. PSS apportions shared pages and is used instead of summed RSS. (d) Nominal capacity of the proposed Threadripper 9960X, 128 GB RAM, and three RTX PRO 4000 Blackwell GPUs relative to the current system. Panel (d) is a hardware-capacity comparison, not measured application speed; aggregate VRAM is distributed across three GPUs and is not pooled. The former 24-step smoke-test VRAM baseline is excluded from this figure.

![購入構成における予測挙動](../results/workstation_predicted_behavior.png)

**Figure 4. Measured baseline and predicted behavior of the proposed three-GPU workstation.** (a) Measured wall-time decomposition of the current 16-environment rollout (960 transitions), separating GPU inference from environment update and other host-side work. (b) Projected aggregate GPU-stage throughput for one to three independent RTX PRO 4000 Blackwell full-height GPUs. (c) Projected end-to-end throughput after applying the measured 41.8% accelerated fraction using Amdahl's law and assigning one independent 16-environment shard to each GPU. (d) Projected capacity of the measured production three-GEM CUDA kernel. (e) Relative completion time for a fixed workload; the current system is normalized to 100%. (f) Nominal workstation capacity relative to the current system. Lines and bars show the planning scenario; shaded envelopes and whiskers span the low-to-high engineering scenarios and are not confidence intervals. All proposed-system results are predictions, not measurements. Aggregate VRAM is distributed across three GPUs and is not pooled.

## 論文用図注

**Figure 1. Computational performance and projected workstation capacity for the three-GEM dFBA reinforcement-learning workflow.** (a) Environment throughput measured with the exact CPU HiGHS backend and the GPU candidate solver on the current workstation. Each 16-environment value represents 960 environment transitions; the 32-environment GPU value represents 640 transitions. (b) Throughput and mean GPU utilization of the production three-GEM CUDA kernel as a function of batch size. Each batch size was measured for 6 s; points are benchmark estimates and connecting lines are visual guides. (c) Nominal hardware-capacity ratios of the proposed workstation relative to the current system. Aggregate VRAM is distributed across three GPUs and is not pooled. (d) Scenario projection for the GPU-only stage. Bars indicate planning estimates; whiskers indicate conservative-to-upper scenarios and are not confidence intervals.

**Figure 2. CPU scaling and proposed sharded execution.** (e) Exact HiGHS label-collection throughput for 96 environment steps, corresponding to 288 LP solves across three GEMs. Each worker count was measured once; the curve therefore describes this benchmark run and does not represent a population mean. (f) Initial deployment target for independent GPU shards. The proposed value is an architecture target, not a measured throughput result.

図中の説明文、購入判断文、ハードウェア仕様表は取り除き、図注へ移した。色はOkabe–Ito系の色覚多様性対応配色とし、白背景、統一線幅、ベクトル文字、グレースケールでも区別できるマーカー・ハッチを使用している。

## 実測値と予測値の区分

| 項目 | 値 | 区分 |
|---|---:|---|
| CPU HiGHS、16環境 | 12.24 transition/s | 現行機実測、960 transitions |
| GPU FBA、16環境 | 85.83 transition/s | 現行機実測、960 transitions |
| GPU化の速度比 | 7.01倍 | 現行機実測 |
| GPU解採用率 | 99.42% | 現行機実測、2,928試行中fallback 17件 |
| GPU FBA最大処理量 | 5,581 prediction/s | 現行機実測、batch 64 |
| GPU使用率 | 平均98.3%、最大100% | 現行機実測、batch 128 |
| batch 128のVRAM | 最大7,935MiB | 現行機実測 |
| 厳密FBAラベル生成 | 最大7.44 LP/s | 現行機短時間実測、4 worker |
| RTX PRO 4000非SFF×1 | 現行GPU段比1.70–2.54倍、計画2.20倍 | 仕様ベース予測 |
| RTX PRO 4000非SFF×3 | 現行GPU段比4.08–7.24倍、計画5.94倍 | 独立shard効率80–95%を含む予測 |
| 3枚の計画処理量 | 約33,150 FBA相当/s | batch 64実測×5.94の予測 |

予測上限は、RTX PRO 4000非SFFの公式メモリ帯域672GB/sとFP32 37TFLOPSをRTX 4060 Laptopの基準値と比較し、小さい側の約2.54倍を採用した。保守値1.70倍、計画値2.20倍を併記し、3 GPUでは独立shardの効率を80%、90%、95%としている。この値はGPU純演算段の容量であり、購入機のend-to-end実測値ではない。

## 構成上の条件

1. TRX50マザーボードは、3枚のGPUを物理的に単スロットで搭載でき、必要なPCIe lane配分・補助電源・冷却を満たすものを選定する。
2. CPU 350WとGPU 145W×3だけで公称785Wとなるため、その他部品と過渡負荷を含む電源・冷却余裕を別途確認する。
3. GPUごとに独立したFBAサービスを起動し、seed・agent・環境群をround-robinまたは固定割当する。
4. batch 128はGPU使用率を最大化するが処理量を落とすため、現行実測ではbatch 64を初期設定とする。
5. GPUサロゲートは探索に使用し、最終評価・論文掲載値はCPU HiGHS等の厳密solverで再計算する。
6. RAM 128GBはGEM構築にも使用し、gapseq／CarveMe、COBRApyによるモデル修正、経路追加、gap filling、培地・交換反応の整合性確認、複数モデルの品質評価を並行実行する。

## 再現用ファイル

- 集計・外挿条件: `results/workstation_procurement_evidence.json`
- 総合図: `results/workstation_procurement_overview.png` / `.pdf` / `.svg`
- CPU・RAM図: `results/workstation_cpu_ram_parallelism.png` / `.pdf` / `.svg`
- Resource Usage図: `results/resource_usage_publication.png` / `.pdf` / `.svg`
- Resource Usage集計値: `results/resource_usage_publication.json`
- 購入構成の予測挙動図: `results/workstation_predicted_behavior.png` / `.svg` / `output/pdf/workstation_predicted_behavior.pdf`
- 予測挙動図の入力・仮定・出力値: `results/workstation_predicted_behavior.json`
- Resource Usage用16/32環境再測定: `results/hardware_resource_parallel_rtx4060.json`
- GPU飽和実測: `results/gpu_saturation_2048_multistream_rtx4060.json`
- 16/32環境スケーリング: `results/parallel_env_scaling_2048_shared_fork_rtx4060.json`
- 16環境長時間比較: `results/parallel_env_16_long_multistream_rtx4060.json`
- 新規CPU測定: `results/exact_fba_collection_scaling_current_cpu.json`
- 新規測定スクリプト: `scripts/benchmark_exact_fba_collection_scaling.py`
- 図の再生成: `scripts/create_workstation_procurement_material.py`
- Resource Usage図の再生成: `scripts/create_resource_usage_publication.py`
- 予測挙動図の再生成: `scripts/create_workstation_prediction_figure.py`

## 公式仕様

- AMD Ryzen Threadripper 9960X: 24 core / 48 thread、最大5.4GHz、350W、4-channel DDR5 RDIMM ECC、PCIe 5.0。
- NVIDIA RTX PRO 4000 Blackwell非SFF: 24GB GDDR7 ECC、672GB/s、37TFLOPS、145W、full-height single-slot。
