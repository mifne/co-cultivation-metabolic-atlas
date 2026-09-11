# GRU初期解予測＋GPU LP補正：初版の実測結果

## 結論

GRU/MLPの実装、RTX 4060 Laptopによる実学習、保存モデルのGPU推論、全変数空間のGPU PDHG補正、元LP証明、CPUとの開発比較まで実行した。既定のPPOやGEMは変更していない。

**GRU導入は作動したが、実GEMでの精度認証・CPU超過速度・環境全体のGPU常駐化は未達。** 失敗解をCPUへ隠れて差し戻したり、ゼロフラックスとして培養状態へ流したりはしない。現経路は実験オプションであり本番学習へ昇格させない。

## 実装

| ファイル | 役割 |
|---|---|
| `src/temporal_lp_model.py` | GRU/MLP、環境ID別hidden、reset、因果性とcheckpoint検査 |
| `src/temporal_lp_data.py` | train-only特徴選択・尺度・PCA、主双対解の圧縮/復号 |
| `src/gpu_pdhg_corrector.py` | FP64全空間PDHG、対角前処理、独立block証明、主双対重み |
| `src/temporal_gpu_lp.py` | GPU推論→復号→GPU補正の接続、認証済み履歴だけの再利用 |
| `scripts/train_temporal_lp.py` | GPU教師付き学習、seed分割、モデル選択・保存 |
| `scripts/benchmark_temporal_lp.py` | CPU/cold/前回認証解/平均/MLP/GRUの因果的LP replay |
| `scripts/diagnose_temporal_compression.py` | CPU正解の圧縮による誤差を分離するoracle診断 |

GRUは入力特徴129、latent32、hidden64、1層、45,664パラメータ。MLPは同じ特徴・latent・hidden幅で12,448パラメータ。パラメータ数は同一ではない。現在入力/差分・前回認証済みlatentから更新を予測する。前回のGPU解が不合格なら、そのlatentは次の正解履歴として使わず消去する。

PyTorchとCuPy間の予測/解の受渡しはDLPackでGPU上の配列を共有する。ただしLP入力正規化・特徴抽出・batch投入・有限性検査の同期・checkpoint判定・COBRA互換の結果読戻しはhostに残る。「GPUで数値補正した」ことと「dFBA全体がGPU内完結した」ことは区別する。

## 教師データと学習

新規seed20294501–16、16環境×60step（各環境12時間）を `training_reference` と明示して採取した。3段階合計2,880 LPはCPU証明と保存読戻し検査に合格。保存先 `results/pf_gru_training16x60_20260905/`。capture231.737秒は圧縮/I/O込み、環境構築29.564秒を別記し、オンライン速度比較には使わない。

exchange段階について最初の12軌跡720点をfit/学習、残る4軌跡240点を開発に使用。尺度、特徴選択、PCAはtrainだけでfitし、train/devのLP入力hash重複は0件。既存の `development_diagnostic_not_training` は学習に転用していない。16本は小規模方式確認用であり、十分な独立条件数とはみなさない。

初版160epochの後、latent成分の尺度をtrainだけで標準化して600epochを実行した。保存先 `results/pf_gru_exchange_rank32_scaled_20260905/`。開発損失が最小のcheckpointを保存し、最終epochを無条件採用していない。

| モデル | 採用epoch | GPU学習秒 | 開発teacher-forced latent MSE |
|---|---:|---:|---:|
| MLP | 336 | 2.40 | 約0.697 |
| GRU | 235 | 2.55 | 約0.761 |

学習秒は当該小規模モデルのoptimizer loopであり、教師生成・読込み・PCA準備を除く。PyTorch 2.11.0+cu130、CUDA13、RTX 4060 Laptopを実際に使用した。開発選択は過去のCPU参照を与えるteacher forcingであり、オンライン精度の証拠ではない。GRUは訓練損失を下げたがこの開発指標でMLPを上回らず、単純なモデル大型化を支持しない。

## 独立した開発LP replay

既存の診断seed20293401–04は学習/開発選択seedと分離している。ただし過去に調査した開発データなので、最終holdoutとは呼ばない。現在のCPU参照解は採点専用で、GPU初期値へ渡さない。各方式は自身が認証した直前解だけを使う。

入力にはCPUで生成された培養状態と上流のmaxmin/aggregate結果が含まれるため、これは**exchange単一段階のLP replay**である。3段階のGPU closed-loop、PHA終点誤差、PPO学習時間は評価していない。CPU対照は4 workerのpersistent HiGHSで、既存の強いdictionary初期化対照とは異なる。初期モデル/ライブラリ準備は別計上、各LPの準備・転送・推論・補正・証明・結果変換は総時間に含む。実行順固定、1回の開発測定で信頼区間は付けない。

`results/pf_gru_exchange_scaled4x8_20260905.json`、4環境×8step、1 LP当たり最大2,048 GPU反復：

| 方式 | 全32 LPの処理秒 | 元LP証明合格 |
|---|---:|---:|
| persistent CPU | 2.08 | 32/32 |
| GPU cold | 5.73 | 0/32 |
| GPU MLP＋補正 | 5.59 | 0/32 |
| GPU GRU＋補正 | 5.66 | 0/32 |

全GPU方式が不合格のため、速度倍率・GPU完結率の達成値にはしない。第8stepの最大主残差はcold約0.40、GRU約1.73で、元基準1e-5から大きく離れている。先頭3step・256反復でも全GPU方式0/12であり、補正回数だけを8倍にしても認証には至らなかった。

## 圧縮と補正のどちらが問題か

`results/pf_gru_exchange_compression_oracle_v2_20260905.json` はstep1/2/41の4環境、12 LPで、**正解そのものをPCAへ投影するoracle診断**を行った。これはオンライン候補生成に使わない。

| 候補 | 元LP証明合格 |
|---|---:|
| CPU参照のx/y | 12/12 |
| rank32で再構成したx/y | 0/12 |
| 再構成x＋正解y | 0/12 |
| 正解x＋再構成y | 0/12 |

再構成の最大主残差714.87、双対違反78.48、相補性gap約37,498。PCAは二乗誤差を小さくするが、元の化学量論・境界・双対条件を保存しない。GRUの予測がこのPCA投影に完全一致しても、そのまま認証されるわけではない。一方、同じ低次元空間内に他の制約適合解が存在しないことまでは証明していない。

大きい双対値は最大約1.09e6で、係数1と5e-5の二項等式が関係する。原行列係数は5e-5〜4,800、行絶対和には約9.38e5倍の幅がある。対角stepは安定条件を満たすが、悪条件性を十分に解消していない。

小さな追試として `tau=tau0/omega, sigma=omega*sigma0` を追加した。安全積は不変、既定omega=1。独立した4×3・2,048反復でomega=.003/.001を試すとcoldの終盤gapは約3.68/1.23まで下がったが、主残差約2.07が残り両設定とも0/12。GRU初期化も0/12だった。結果は `pf_gru_exchange_weight003_4x3_20260905.json` と `pf_gru_exchange_weight001_4x3_20260905.json`。これ以上の無差別なweight sweepや反復増加は行わない。

## 次の構造変更

1. **制約保存型decoder**を優先する。固定Sの独立等式/零空間表現、固定ゼロ列や二項等式の厳密消去をofflineで検証し、原LPへの復元と双対復元まで証明する。主双対を無制約にPCA回帰する方式は比較対照として残す。
2. GRUは自由変数の更新やactive-set切替の予測に使い、主双対を別に扱う。現在CPUが偶然選んだ多重最適解へのMSEだけでなく、固定回数補正後の元LP残差と補正費用を学習/選択指標にする。
3. 小規模・中盤で認証できた構成だけ、全3段階GPUの因果的rolloutへ接続する。CPU生成の上流結果を使うreplayを、その代替にしない。
4. その後にdevice状態更新、host同期削減、環境数拡大、120step、強いCPU対照との反復比較へ進む。最終holdoutは構成固定後に新たに採取する。

## 検証と環境制約

従来経路＋PDHGは704件、新しいGRU/codec/接続/診断は別processで71件、計775件が合格した。GPU処理中にCPU最適化を禁止するtoy試験、未収束時のcache破棄、env ID/reorder/reset、未来入力非参照、checkpoint不整合拒否を含む。

全てを同一pytest processへ混在させると54件失敗・714件合格だった。新しいPyTorchの早期importと既存CuPyのCUDA探索が組み合わさると、CUDA13 wheelのrootから既存conditional graph用 `libcudadevrt.a` を見つけられない。加えて既存GPU割当テストが環境変数を残す問題を修正した。現在は従来cuOpt/graphと新しいTorch/PDHG試験を別processに分離し合格を確認した段階で、ライブラリ互換性の恒久解決ではない。依存ライブラリの置換・upgradeや失敗試験のskip追加はしていない。

初回replayのJSON出力時にCPU診断内のinfinityで保存エラーが出たため、既存の非有限値→null変換を適用してv2以降を再実行した。初回ファイルはfailedと明示して保存し、速度/精度結果として使っていない。生データ・source snapshots・失敗案は削除していない。現在の計測プロセスはすべて終了している。
