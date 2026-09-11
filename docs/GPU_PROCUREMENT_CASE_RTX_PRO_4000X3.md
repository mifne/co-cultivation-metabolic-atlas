# RTX PRO 4000 Blackwell SFF ×3 導入判断資料

作成日: 2026-08-25  
対象: 3 GEM dFBA 強化学習（OR16、NS21、*L. plantarum*）

## 発注意思（そのまま転記できる文案）

3種GEMを用いるdFBA強化学習では、CPU HiGHSによるFBAが1ステップ時間の
89%を占めていた。RTX 4060 Laptop上の物理制約付きGPUサロゲートへ置換した
受入試験では、同一24ステップを5回ずつ比較して全体実行時間が平均13.05秒から
2.63秒へ短縮し、ペア比較で4.99±0.40倍（95%信頼区間）の高速化を確認した。

次段階では、単一環境の待ち時間短縮ではなく、複数環境・複数seed・複数agentを
GPU単位に分割して総探索量を増やす。RTX PRO 4000 Blackwell SFFは1枚24GB、
432GB/s、70W、非SFF版は24GB、672GB/s、140Wである。RTX 4060実測と公式仕様から、
3枚の純GPU FBA処理容量は現行1枚比でSFF版4.19倍、非SFF版6.52倍、3系統rolloutは
それぞれ2.62倍、2.66倍と見積もる。大規模バッチを本番要件とする場合は、
RTX PRO 4000 Blackwell 非SFF版 ×3の発注を希望する。

## 図の読み方

![RTX PRO 4000 ×3 導入判断グラフ](../results/gpu_procurement_case_rtx_pro_4000x3.png)

- A: 同じ24ステップをCPUとGPUで各5回実行した全体時間。GPU化自体の効果を示す。
- B: CPU版ではFBAが89%を占める。GPU版ではFBA比率が50%まで下がり、前処理も
  同程度となった。これは「CPU FBA律速をGPUで解消した」証拠である。
- C: RTX 4060で同時環境数を1から64へ増やすと、安全検査込み処理量は
  OR16で5.94倍、NS21で1.75倍、*L. plantarum*で5.18倍となった。
- D: SFF版と非SFF版を同じ尺度で比較。非SFF版は純GPU容量で約56%高い一方、
  現行規模のrollout差は約2%にとどまり、消費電力は2倍となる。

## 重要な判断条件

現時点の102候補辞書はディスク上6.8MBで、単一環境ではRTX 4060のVRAMを
使い切っていない。またGPU版の残り時間の多くはCPU前処理とHiGHS fallbackである。
したがって「現在の単一環境がGPU不足」という説明は実測と一致しない。

3枚の根拠は次の運用を実施することにある。

1. 1 GPUにつき1つのGPUサロゲートサービスを起動する。
2. 環境workerを `--gpu-ids 0,1,2` でround-robin分配する。
3. 3枚を1つのCUDAメモリとして扱わず、seed・agent・環境群を独立shardにする。
4. 本番は16環境以上を目安にし、各GPU内で最大64要求をmicro-batchする。
5. 最終評価値はCPU HiGHSで再計算し、GPUサロゲートは探索に限定する。

この運用をしない場合、1枚をRTX 4060からRTX PRO 4000へ替えても、単一rolloutの
予測短縮は約1.03倍にとどまる。3枚購入は「単一runの高速化」ではなく、
「3系統を同時に走らせる探索スループットと再現実験の並列化」の投資である。

## 実測条件と外挿式

| 区分 | 値 | 扱い |
|---|---:|---|
| CPU HiGHS 24-step | 13.05 ± 0.56秒 | RTX 4060機上の実測、n=5、95% CI |
| GPU surrogate 24-step | 2.63 ± 0.24秒 | RTX 4060機上の実測、n=5、95% CI |
| ペア速度比 | 4.99 ± 0.40倍 | 実測、n=5、95% CI |
| GPU安全採用率 | 91.4–98.4% | 3 GEM holdout受入試験 |
| PRO 4000単体の純GPU比 | 1.64倍 | `min(432/256, 24/14.6)` |
| SFF×3 純GPU容量 | 3.95–4.93倍 | GPU間効率80–100%、計画85%=4.19倍 |
| 非SFF×3 純GPU容量 | 6.14–7.67倍 | SFF比の帯域/FP32外挿、計画85%=6.52倍 |
| SFF×3 総rollout | 2.46–3.08倍 | Amdahl補正＋GPU間効率、計画85%=2.62倍 |
| 非SFF×3 総rollout | 2.50–3.13倍 | Amdahl補正＋GPU間効率、計画85%=2.66倍 |
| VRAM | 両者24GB/枚、72GB合計 | 非共有。各shardは24GB上限 |
| GPU消費電力 | SFF 210W / 非SFF 420W | 3枚合計、非SFFは2倍 |

FP32 24 TFLOPSと帯域432GB/sのうち小さい倍率を使ったため、AI TOPS比は外挿に
使用していない。総rollout予測では、現行GPU実測のサロゲート推論時間だけを
1.64倍し、CPU前処理・fallback・状態更新は短縮しないAmdahlモデルとした。

## 根拠ファイル

- 5回反復の生ログ: `results/rollout_cpu_vs_gpu_rtx4060.json`
- GEM別バッチ実測: `results/fba_surrogate_gpu_lp_rtx4060.json`
- 図の集計値・外挿条件: `results/gpu_procurement_case_rtx_pro_4000x3.json`
- 提出用一枚図: `results/gpu_procurement_case_rtx_pro_4000x3.pdf`
- 再生成: `python3 scripts/create_gpu_procurement_material.py`
- 再ベンチマーク: `python3 scripts/benchmark_rollout_cpu_gpu.py --steps 24 --repeats 5`

## 仕様出典

- [NVIDIA RTX PRO 4000 Blackwell SFF製品ページ](https://www.nvidia.com/ja-jp/products/workstations/professional-desktop-gpus/rtx-pro-4000-sff/)
- [NVIDIA RTX PRO 4000 Blackwell 非SFF製品ページ](https://www.nvidia.com/ja-jp/products/workstations/professional-desktop-gpus/rtx-pro-4000/)
- [NVIDIA公式データシート](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-4000-blackwell/workstation-datasheet-blackwell-rtx-pro-4000-gtc25s-nvidia-3662515-r6.pdf)
- [NVIDIA GeForce RTX 40 Laptop GPU仕様](https://www.nvidia.com/en-eu/geforce/laptops/40-series/)
