# NVIDIA cuOpt FBA バックエンド統合ドキュメント

**統合文書の要約**: 本ドキュメントは、シミュレータにおけるNVIDIA cuOptを利用したFBA（束縛フラックス解析）バックエンドの実装、評価、および導入方法を包括的に記述する。複数の試行・検証記録を統合し、cuOptの有効性、性能特性、および現状の限界を提示する。GPU環境におけるdFBA/LPソルバーの開発進捗と、RTX 4060を用いた実測に基づく評価結果を中心に構成する。

## 1. 導入：cuOptをFBAバックエンドとして選択する理由

FBAの各呼び出しは連続線形計画問題であり、制約が`S v = 0`、反応範囲と単一線形目的関数を持つ形式である。NVIDIA cuOptはPython APIを通じてLPを公開し、CSR制約行列、GPU上でのPDLP/バリア解法、変数範囲更新、および基本解へのクロスオーバーをサポートする。後者は、デフォルトのPDLP解が必ずしも頂点解でないため、FBAにとって重要である。

アダプタは各種の化学量論行列を初回に構築し、dFBAの各ステップで範囲と目的係数を更新する。完全LPを毎ステップで再構築しない。

反復解法のパスでは以下の最適化を実装：
- 変更された反応範囲/目的係数だけを遅延してcuOpt問題に送信
- ソルごとのコンソールログを無効化
- `max(abs(S @ v))`と反応範囲違反をcuOptの報告ステータスから独立して検証
- ピポットQRによる数値従属化学量論行の除去（戻りのフラックスは完全な元の行列で検証）
- 最初のGPU目的を同一符号付きフラックスのHiGHS LPと監査
- 失敗したBarrierソルをランク削減PDLPで再試行
- 可能な場合、PDLP後にクロスオーバーを使用、クロスオーバー自身が退化ゼロ目的状態で失敗した場合のみ再試行せず
- そのステップに過渡的な時間制限をCPU LPへ送信、LP失敗を生物消失として扱わない

(出典: `GPU_CUOPT.md`)

## 2. RTX 4060 でのベンチマーク結果（2026-08-25）

### 2.1 初期問題と修正

元のGPU有効性0%はCUDA実行不足ではなく、複数の相互作用する問題による：
1. 最初のアダプタ版ではcuOptソルブにCOBRA最大化方向を強制できず
2. 固定反応を`[0, 0]`から`[-1e-6, 0]`に変更し、可能領域を変異しゴーストフラックスを発生
3. GEM化学量論行列に従属保存行が含まれており、バリア/クロスオーバーを不安定化、PDLPを低速化
4. クロスオーバーをグローバルに無効化するとcuDSSエラーは回避されるが、不正確な内部PDLP解を残す
5. 一度の時間制限が種を永久に無効化する問題

ピボテッドQRでOR16から39行、NS21から31行、LPから52行の従属行を除去。フル`S @ v`は受容判定に使用するため、行列削減が質量バランスを隠して緩くしない。

### 2.2 修正後のベンチマーク（3ステップ6種、3 GEM/ステップ）

| フェーズ | 平均秒/ステップ | シェア |
|---|---:|---:|
| 動的な範囲/目的準備 | 0.0527 | 9.5% |
| 3FBA解決 | 0.4973 | 89.9% |
| 環境/状態更新 | 0.0027 | 0.5% |
| 合計 | 0.5527 | 100% |

修正後、3ステップのGPU検証で9/9 LPがcuOptで完了。6ステップ間で17/18のGPU試行が受け入れられ、1ステップ（OR16）が時間制限にヒットし、そのステップのみCPU正確フォールバックを使用。どの種も永続的に無効化されなかった。

| パス | ステップ/秒 | 有効GPU LP率 | エンドツーエンド有効結果 |
|---|---:|---:|---:|
| GLPK正確範囲 | 1.808 | n/a | 100% |
| ランク削減PDLP (3 steps) | 0.195 | 100% | 100% GPU |
| ランク削減PDLP (6 steps) | 0.191 | 94.4% | 85.7% ※ |

※ 1 CPUフォールバックにより100%の有効状態が保たれた。

### 2.3 正確性検査

同一の3ステップ比較で、CPUとGPUの最終状態ベクトルは同一（`max_abs_diff = 0.0`）、GPU有効性は9/9、フォールバックなし。これにより、RLエージェントが実際に観測する状態の正確性が確認された。

### 2.4 結論：cuOptの加速度効率

正確性は修復されたが、単一環境でのGPU加速度はRTX 4060では達成されず：PDLP反復とクロスオーバーによりGLPK（正確範囲）より約9.4倍遅い。修正後のバッチ解決実験でも4～8環境で約0.33環境ステップ/秒で飽和し、コネクション競合で時間制限が増加。cuOptのバッチAPIは現在のカードでは速度も信頼性もトレーニングパスに統合するのに十分ではない。

## 3. 3-GEM連結計算（`--fba-mode joint`）

`--fba-mode joint`はOR16、NS21、LPの3つのGEMからブロック対角疎LPを構築する（6,137変数、4,117制約行、25,932非Zero要素 for 最終コンソーシアム）。これは3つの独立FBA目的と等価でありながら、各環境のdFBA状態更新での共有媒体結合を保持する。安全な性能実験であり、新しいOptM様の負荷目的ではない。

CPU連結パスはSciPy/HiGHSを使用。現在のRTX 4060/cuOpt 26.2では、この高度非連結ブロック行列に対してBarrier+クロスオーバーがcuDSS CSRエラーを報告することがある。アダプタは失敗時にPDLP（クロスオーバーなし）で再試行し、`concurrent`は安全なPDLPパスに再マップ。明示的な`cuopt`モードは両GPU試行が失敗した場合のみ大きなエラーを表示。GPU接尾は約2.8～3.3秒/ステップ（初期化含む）、CPU連結HiGHSは約0.33秒/ステップ。PDLP残差はプロジェクトの科学誤差と比較して確認必要。

(出典: `GPU_CUOPT.md`)

## 4. 設置と実行方法

### 4.1 cuOptの設置

cuOptのホイールをCUDAランタイムおよびドライバにあわせて設置。NVIDIAの設置選択で、CUDA 13は`cuopt-cu13`、CUDA 12は`cuopt-cu12`を使用：

```bash
# CUDA 13対応
python -m pip install --extra-index-url=https://pypi.nvidia.com cuopt-cu13

# 必要に応じてバージョン固定（例: cuopt-cu13==26.2.*）
# 実用では互換性確認後固定
```

パッケージを基本要求依存に追加しないこと：オプションのNVIDIAホイールであり、デフォルト開発環境では利用できない。

(出典: `GPU_CUOPT.md`)

### 4.2 実行方法

```bash
# 明示的なGPUプローブ；無効な種の解決はGLPKに切り替え
DFBA_SOLVER=cuopt python main.py train --sbml-dir models/sbml/final_consortium \
  --solver-backend cuopt --fba-mode separate --cuopt-method pdlp

# 三モデル連結の大きなCPU LP（主要確認用）
python main.py train --sbml-dir models/sbml/final_consortium \
  --solver-backend glpk --fba-mode joint

# cuOptを調査し、安全にGLPKにフォールバック
DFBA_SOLVER=auto python main.py train --solver-backend auto

# 各GPUに固定的に環境を割り当て
DFBA_SOLVER=cuopt DFBA_GPU_IDS=0,1,2 python main.py train \
  --solver-backend cuopt --fba-mode separate \
  --gpu-ids 0,1,2 --n-envs 3 --gpu-slots-per-device 1 --device cpu
```

1つのGPUで複数環境を共有する場合（別々のCUDA/cuOptコンテキストを使用）：

```bash
DFBA_SOLVER=cuopt DFBA_GPU_IDS=0 python main.py train \
  --solver-backend cuopt --fba-mode separate \
  --gpu-ids 0 --gpu-slots-per-device 4 --n-envs 4 --device cpu
```

これは協力的プロセスレベル並行であり、MIGやハードVRAM分割ではない。環境数に対してベンチマークし、過剰割り当てのスループットロスを確認すること。

### 4.3 RTX 4060 拡張性プロット（参考）

PPOベンチマーク（全64時間ステップ、個別FBA、CPU PPO、1 GPU cuOpt）。CPUはGLPKを使用：

| 環境数 | CPU GLPK (step/s) | GPU cuOpt (step/s) | GPU/CPU |
|---:|---:|---:|---:|
| 4 | 7.46 | 5.83 | 0.78x |
| 8 | 10.46 | 6.91 | 0.66x |
| 16 | 6.92 | 4.01 | 0.58x |

GPUスループットは8ワーカー付近でピークだが、このモデル/ハードウェアではCPUを上回らなかった。これらの古いスループット数値は残差監査前であり、確認済みの解決スループットとして解釈してはならない。

(出典: `GPU_CUOPT.md`)

## 5. ベンチマークプロトコル

同じシードと短い範囲で、`--solver-backend glpk`、`cuopt`、`auto`を使用して、壁時間、`time/fps`、目的値、最終状態指標を記録する。GPU結果は、目的/フラックス差がプロジェクトの科学誤差範囲内で、かつエンドツーエンド壁時間が改善された後でのみGLPKを置き換えるべき。

(出典: `GPU_CUOPT.md`)

## 6. GPU常驻カーネルチェック

`scripts/benchmark_gpu_resident.py`を使用して、タイミング済みスパース行列セクションの間、化学量計量CSR行列と状態ベクトルをCUDA上に保持。RTX PRO 4000 SFFに対するAmdahl型推定をパブリックバンド幅/FP32最大値に基づいて報告。これはハードウェアスケーリング推定だが、実際のcuOpt LP解決の代わりにならない。

## 7. 並列PPOベンチマーク

`scripts/benchmark_ppo_devices.py`を使用して、完全並列dFBA/PPOパスをCPU vs CUDAで比較。環境ワーカーは依然としてCPUのCOBRApy/GLPKを使用。`--device cuda`はPPOのMLP/値ネットワークをGPUに移動。実際のcuOpt実行では`--solver-backend cuopt`を使い、各ワーカーを`CUDA_VISIBLE_DEVICES`でマッピングする。

(出典: `GPU_CUOPT.md`)

## 8. ディスカッション・注意点

現在のcuOptバックエンドの性能制約から、RTX 4060でのGPU加速度は達成されていない。RLトレーニングパイプラインにはcuOptを統合せず、ワークフローはGPUのPPOとCPUのFBAを分離して実行することを推奨する。cuOptの速度と信頼性が向上した場合にのみ、統合すべきである。

**見逃し防止の注意**：NVIDIA cuOptのバンドルLPAPI（http://docs.nvidia.com/cuopt/user-guide/latest/lp-qp-features.html）と最適化ソルバ設定（https://docs.nvidia.com/cuopt/user-guide/26.02.00/lp-qp-milp-settings.html）の基準から、PDLPの低精度バリアへの影響、クロスオースタのトレードオフ、LPバッチモードの廃止などが報告されている。

## 構成元ファイル

- `GPU_CUOPT.md` (GPU_CUOPTドキュメント) — 複数ソースを含む統合ファイルで、HTML生成基準をHGPで確認可能。なお、提供されたファイルは1つであるが、統合先の多重ファイルの場合は複数列挙する必要がある。本統合文書はこの1ファイルに基づく。
