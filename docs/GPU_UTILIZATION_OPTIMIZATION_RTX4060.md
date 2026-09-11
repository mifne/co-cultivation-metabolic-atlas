# RTX 4060 GPU利用率改善・CPU比較

## 測定対象

- Windows表示: GPU 1（NVIDIA GeForce RTX 4060 Laptop GPU）
- WSL/PyTorch表示: `cuda:0`
- PCI bus: `00000000:01:00.0`
- Intel内蔵GPUはCUDAデバイスとして列挙されず、本測定には使用していない。

## 公平な比較条件

- CPU版とGPU版へ同じ乱数seed・同じaction tensorを入力した。
- 各条件は12 vector steps、環境数2・4・8・16で測定した。
- モデル読込、CUDA初期化、最初のwarm-up stepは計測から除外した。
- throughputは`環境数 × vector steps / 測定秒数`で算出した。
- GPU使用率・VRAMは`nvidia-smi`、CPU使用率は`psutil`で0.2秒間隔に取得した。
- CPU版は厳密HiGHS、GPU版は物質収支・境界条件guard付きCUDA候補辞書である。GPU版で有効候補がない場合はHiGHSへfallbackする。
- 最終的な科学評価は厳密HiGHSで行い、GPU版はPPO rollout高速化に使用する。

## 実装した改善

1. CUDAサービスを`fork`から`spawn`起動へ変更し、親プロセスがCUDAを初期化した後でもGPUサービスを開始できるようにした。
2. 境界違反・物質収支残差のguardをサンプルごとのSciPy処理からGPUテンソル一括処理へ変更した。
3. 1環境から3 GEMを3回のRPCで送る構成を、1回の`predict_many`で同時投入する構成へ変更した。
4. GPUサービスへbatch size histogramとGEM別batch統計を追加した。
5. 候補辞書方式では学習分布距離が妥当性判定にならないため、物理guardを維持したまま誤ったOOD fallbackを除外した。
6. 厳密解候補を102件から512件へ拡張し、validation用に除外されていた候補も実行可能解辞書へ含めた。
7. 同一入力によるCPU/GPUスケーリング、GPU/VRAM/CPU監視、出力差、fallback率を同時記録するベンチマークを追加した。
8. 候補辞書を2048件へ拡張し、厳密HiGHSラベル収集を4 CPUプロセスへ並列化した。
9. GPU管理プロセス内の逐次的なpandas／COBRA `Solution`生成を廃止し、raw配列を各環境ワーカーへ返してCPU後処理を分散した。
10. 3 GEMを別々のCUDA streamで同時実行し、候補評価・境界guard・物質収支guardの重なりを増やした。
11. 3 GEMを親プロセスで1回だけ読み込み、Linux `fork`のcopy-on-writeで環境ワーカーへ共有した。これにより、WSLが認識する約23 GiB内でも32環境が完走した。

## RTX 4060実測結果（512候補版）

| 並列環境数 | CPU HiGHS (transitions/s) | CUDA (transitions/s) | CPU比 | GPU平均 | GPU P95 | GPU最大 | VRAM最大 | GPU解受理率 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3.79 | 18.64 | 4.92× | 7.4% | 14.0% | 15% | 161 MiB | 96.2% |
| 4 | 6.08 | 29.35 | 4.83× | 8.4% | 16.0% | 19% | 159 MiB | 96.2% |
| 8 | 8.86 | 40.96 | 4.62× | 11.4% | 19.1% | 20% | 181 MiB | 96.5% |
| 16 | 11.97 | 46.56 | 3.89× | 13.0% | 24.5% | 29% | 287 MiB | 97.1% |

修正前のGPU平均使用率は環境数2・4・8・16で3.7%、3.8%、5.9%、4.9%だった。512候補版では7.4%、8.4%、11.4%、13.0%となり、16環境では平均使用率が約2.6倍になった。

## 解釈

- GPU使用率とGPU解受理率は明確に改善した。16環境ではCUDA rolloutがCPU/HiGHSの3.89倍である。
- 102候補版は16環境で63.34 transitions/sに達し、512候補版より速い。一方、512候補版はGPU解受理率とGPU使用率が高い。GPU使用率だけを最大化すると候補比較量が増え、end-to-end throughputを落とすため、利用率と処理速度は分けて評価する必要がある。
- 16環境でもVRAMは287 MiBであり、現在の律速はVRAM容量ではない。環境更新、COBRAモデル操作、fallbackによるworker非同期化が残るCPU側律速である。
- RTX PRO 4000級GPUを有効利用するには、さらに環境数を増やすだけでなく、環境状態更新と特徴量生成を配列化し、GPUへ継続的に大きなbatchを供給する必要がある。

## 最新実測（2048候補・raw payload・3 CUDA streams）

連続する実FBA候補評価だけを測る飽和試験では、ダミー行列演算や無意味なVRAM確保を一切加えず、3 GEMの候補最適化と物理guardを実行した。

| GEMごとのバッチ | GPU平均 | GPU最大 | VRAM | 処理量（3-GEM予測/s） | 判定 |
|---:|---:|---:|---:|---:|---|
| 8 | 56.9% | 63% | 621 MiB | 3,179 | 小規模 |
| 16 | 69.0% | 73% | 1,131 MiB | 4,156 | 現行CPU上の実用域 |
| 32 | 76.7% | 83% | 2,187 MiB | 4,961 | スケール継続 |
| 64 | 84.3% | 89% | 4,331 MiB | **5,581** | **処理量最適** |
| 128 | **98.3%** | **100%** | **7,935 MiB** | 1,299 | VRAM圧迫で低速化 |

バッチ128ではRTX 4060 Laptopの8GB VRAMとGPU演算器をほぼ使い切ったが、メモリ圧迫により処理量が低下した。したがって、GPU使用率100%は達成可能でも、運用推奨値ではない。現行GPUの推奨点はバッチ64である。

16環境・960遷移の長時間end-to-end比較では、CPU HiGHS 12.24 transitions/sに対しCUDA版85.83 transitions/sで、**7.01倍**だった。GPU解受理率は99.42%、CPU fallbackは17/2,928 solve attemptsだった。end-to-endのGPU平均は6.8%である。これはGPU計算が短いburstで終了し、残り時間をCOBRAモデル更新・環境状態更新・プロセス間通信が占めるためで、GPUカーネル自体の飽和値とは区別する必要がある。

## RAMの確認

Resource Usageの論文用図は、実FBAカーネルのGPU飽和、VRAM圧迫とthroughput、16/32並列環境のsystem CPUとprocess PSS、購入構成の公称容量比で再構成した。短時間24-step smoke testのVRAM値は使用していない。図は`results/resource_usage_publication.png`、`.pdf`、`.svg`、集計値は`results/resource_usage_publication.json`、16/32環境の再測定値は`results/hardware_resource_parallel_rtx4060.json`、再生成スクリプトは`scripts/create_resource_usage_publication.py`である。

- Windows搭載物理RAMは48GB（24GB DDR5-4800 ×2）。
- 現在のWSLは既定上限のため約23GiBを認識している。48GB搭載量そのものが23GBという意味ではない。
- copy-on-write共有後は32環境が現行WSL上限内で完走したが、16論理CPUを超える32環境ではCPU側が過剰並列となり、CUDA throughputは90.6から76.2 transitions/sへ低下した。
- Threadripper 9960XではCPUコア数が増えるため、64前後の環境バッチをGPUへ継続供給しやすくなる。複数GPUの主用途は、GPUごとに独立した環境群・seedを割り当てる総スループット向上である。

## 科学的な注意

GPU版は厳密LPそのものではなく、厳密HiGHS解から作った実行可能候補辞書を探索する近似rollout backendである。物質収支と境界条件はguardするが、候補集合にない最適点との差は残る。60ステップのCPU/GPU軌跡差は蓄積するため、最終結果・論文値はCPU HiGHSまたはGPU解の厳密監査で確定する。GPU使用率やVRAM使用量だけを購入根拠にせず、7.01倍のend-to-end処理量、99.42%のGPU解受理率、複数seed同時実行能力と併記する。

## 再現コマンド

```bash
python3 scripts/benchmark_parallel_env_scaling.py \
  --artifact-dir models/fba_surrogate_gpu_lp_512 \
  --env-counts 2 4 8 16 \
  --vector-steps 12 \
  --interval 0.2 \
  --batch-window-ms 4.0 \
  --output results/parallel_env_scaling_512_rtx4060.json \
  --plot results/parallel_env_scaling_512_rtx4060.png

python3 scripts/create_gpu_utilization_improvement_report.py

python3 scripts/benchmark_gpu_saturation.py \
  --artifact-dir models/fba_surrogate_gpu_lp_2048 \
  --batch-sizes 8 16 32 64 128 \
  --output results/gpu_saturation_2048_multistream_rtx4060.json \
  --plot results/gpu_saturation_2048_multistream_rtx4060.png

python3 scripts/create_gpu_saturation_material.py
```
