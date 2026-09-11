# プロジェクト・ドキュメント・マスターインデックス

本プロジェクト（天然ゴム分解・PHA変換コンソーシアム）の全記録を、整理・集約された以下の項目で管理しています。

## 🗺️ 1. プロジェクト管理・戦略 ([PROJECT_MANAGEMENT.md](PROJECT_MANAGEMENT.md))
- プロジェクトの変遷史（4種→3種(LP)→3種(Pf)への進化）。
- コンソーシアム（OR16, NS21, Pf）の選定根拠と役割分担。
- 実装ロードマップと現在の開発ステータス。

## 🧬 2. 代謝モデル・インデックス ([MODEL_INDEX.md](MODEL_INDEX.md))
- **【最重要】** 各微生物モデルの由来（ゲノム）と改修履歴。
- 改修スクリプト（Lcp/Rox経路の実装）との紐付け。
- 科学的整合性の検証ステータス。
- 現行の第3菌種(P. freudenreichii)と不採用となったL. plantarumの経緯。

## 🧪 2.5 3種共存の生物学的知見・実験計画
- [OXYGEN_SCHEDULE_THIRD_SPECIES_20260910.md](OXYGEN_SCHEDULE_THIRD_SPECIES_20260910.md): 酸素供給条件と3種価値の研究(最新・最も整理された比較)。
- [THREE_SPECIES_COEXISTENCE_SUMMARY_20260911.docx](THREE_SPECIES_COEXISTENCE_SUMMARY_20260911.docx): 3種共存に関するこれまでの知見・新規性評価・校正パラメータの統合報告書。
- [CALIBRATION_EXPERIMENT_PROTOCOL_20260911.docx](CALIBRATION_EXPERIMENT_PROTOCOL_20260911.docx): 未校正パラメータを実測するための湿式実験プロトコル。

## ⚙️ 3. システム・アーキテクチャ ([SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md))
- 多段階代謝シナジー（Lcp $\rightarrow$ Rox $\rightarrow$ PHA）の物理的定義。
- dFBA-RL 統合制御ループの仕組み。
- 質量収支および科学的妥当性の証明。
- [cooperative_gpu_surrogate.md](cooperative_gpu_surrogate.md): 複数環境を束ねるGPU FBA辞書、厳密フォールバック、RTX 4060受入試験、Pf構成での認定状況(2026-09-11時点で未認定)。
- GPU加速dFBA/LPソルバー開発の詳細記録は2026-09-11に57件から15件へ統合された。一覧は本書末尾の「GPU高速化関連ドキュメント」を参照。

## 🧪 4. 科学的評価基準 ([RL_EVALUATION_PLAN.md](RL_EVALUATION_PLAN.md))
- 強化学習環境および dFBA シミュレーターの科学的妥当性をトップジャーナル水準で評価するための厳密な計画と基準。

## 📂 5. ディレクトリ構造とコード管理
プロジェクトの保守性を高めるため、以下の体系でコードを管理しています。

- **`src/`**: シミュレータ、環境、エージェントのコアロジック。
    - `utils.py`: モデル読み込みや初期化設定の共通ライブラリ。
- **`scripts/`**: 目的別に分類されたユーティリティ群。
    - `analysis/`: エージェント評価、理論限界、時系列データの分析。
    - `automation/`: 学習、最適化、タスク実行の自動化。
    - `debug/`: 代謝経路診断、ソルバー動作確認、単体テスト用。
    - `model_ops/`: SBMLモデルの改修・更新スクリプト。
    - `validation/`: 長期シミュレーションと科学的整合性の検証。
- **`outputs/`**: 実行結果の集約先。
    - `logs/`: トレーニングおよびデバッグログ。
    - `checkpoints/`: 学習済み強化学習モデル。
- **`tests/`**: Pytestによる自動検証スイート。整合性維持の要。

---

## 📊 最終成果・証明資料
- **[FINAL_CONSORTIUM_REPORT.md](FINAL_CONSORTIUM_REPORT.md)**: 学習結果の要約、分解率・PHA収率の最終報告およびシミュレーションデータ詳細。
- **[FINAL_MASS_BALANCE_PROOF.png](FINAL_MASS_BALANCE_PROOF.png)**: 質量収支誤差 0.1% 未満を証明する決定的なグラフ。
- **[ULTIMATE_SYSTEM_PROOF.png](ULTIMATE_SYSTEM_PROOF.png)**: 代謝シナジーと AI 制御を多角的に可視化した最終証明図表。

## ⚙️ 6. GPU高速化関連ドキュメント(2026-09-11に57件→15件へ統合)

GPU加速dFBA/LPソルバー開発の詳細記録。いずれも「計画→結果」または同一テーマの複数回試行を時系列で統合し、末尾に構成元ファイルを明記している。

- [cooperative_gpu_surrogate.md](cooperative_gpu_surrogate.md): 辞書ベースGPUサロゲート(32,768候補辞書、ニューラル順位付け、QP投影)。CPU比2.6〜2.8倍高速だがPHA精度1%基準は未達成。
- [gpu_hybrid_lp.md](gpu_hybrid_lp.md): 検証付きGPU基底辞書+CPU再最適化のハイブリッド方式。
- [gpu_batch_speed_optimization.md](gpu_batch_speed_optimization.md): GPUバッチ内点法のホットタイム最適化試行。CPU超えは未達。
- [gpu_ipm_stability.md](gpu_ipm_stability.md): GPU内点法の数値安定性修正とCPU比較。
- [dataflow_acceleration.md](dataflow_acceleration.md): dFBA全体のデータフロー律速分析と非同期パイプライン化。
- [gnn_gru_supervised_correction.md](gnn_gru_supervised_correction.md): GNN+GRUによるLP初期解予測。前解方式を上回れず。
- [compact_gpu_acceleration.md](compact_gpu_acceleration.md): コンパクト基底辞書/低次元制限表によるGPU LP解法。
- [amn_guided_acceleration.md](amn_guided_acceleration.md): AMN(Faure et al. 2023)に触発されたニューラル初期基底選択。
- [gpu_literature_refinement.md](gpu_literature_refinement.md): 論文知見に基づくGPU機構層改良と目的関数忠実度診断。
- [ppo_accuracy_speed.md](ppo_accuracy_speed.md): PPO学習に必要な精度とGPU数値層の安定収束の分離設計。
- [cpu_baseline_fairness.md](cpu_baseline_fairness.md): GPU速度主張の前提となるCPUベンチマーク条件の監査。
- [gpu_device_qualification.md](gpu_device_qualification.md): デバイスピボット+LU再利用版GPUソルバーの回帰検証。
- [gpu_speed_priority_plan.md](gpu_speed_priority_plan.md): 高速化優先順位変更の経緯と次期アーキテクチャ設計。
- [gpu_cuopt_backend.md](gpu_cuopt_backend.md): NVIDIA cuOptをFBAバックエンドとして使うアダプタ設計と既知の制約。
- [gpu_stream_partition.md](gpu_stream_partition.md): CUDAストリーム分割の設計案(実装・実測は未実施)。

一部の試行は限定条件下でCPU HiGHSより高速(例: cooperative_gpu_surrogateの狭い条件で2.6〜7倍)だが、**いずれも科学的認定基準(24時間PHA相対誤差1%以下等)を満たしておらず、2026-09-11時点で最終評価・本番学習に昇格したGPU近似は存在しない**(常にCPU HiGHSへフォールバック)。

根本原因は「後段CPU完了待ち」(GPU計算後のホスト側dFBA状態更新・認証)が全体時間の過半(データフロー解析で54.3%)を占めていたこと。改善方針は[CPU_BOTTLENECK_IMPROVEMENT_PLAN_20260911.md](CPU_BOTTLENECK_IMPROVEMENT_PLAN_20260911.md)を参照。

---
**現在の状況**: OR16+NS21の2種基盤開発は完了し、P. freudenreichiiを加えた3種共存の
価値評価・GPU加速dFBA基盤の構築・湿式実験での校正に向けた準備を並行して進めています。
変更を加えた際は、必ず `pytest tests/test_project_integrity.py` を実行し、整合性を担保してください。
