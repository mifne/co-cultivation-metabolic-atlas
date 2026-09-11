# プロジェクト・ドキュメント・マスターインデックス

本プロジェクト（天然ゴム分解・PHA変換コンソーシアム）の全記録を、整理・集約された以下の項目で管理しています。

## 🗺️ 1. プロジェクト管理・戦略 ([PROJECT_MANAGEMENT.md](PROJECT_MANAGEMENT.md))
- プロジェクトの変遷史（4種から3種への進化）。
- コンソーシアム（OR16, NS21, LP）の選定根拠と役割分担。
- 実装ロードマップと現在の開発ステータス。

## 🧬 2. 代謝モデル・インデックス ([MODEL_INDEX.md](MODEL_INDEX.md))
- **【最重要】** 各微生物モデルの由来（ゲノム）と改修履歴。
- 改修スクリプト（Lcp/Rox経路の実装）との紐付け。
- 科学的整合性の検証ステータス。

## ⚙️ 3. システム・アーキテクチャ ([SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md))
- 多段階代謝シナジー（Lcp $\rightarrow$ Rox $\rightarrow$ PHA）の物理的定義。
- dFBA-RL 統合制御ループの仕組み。
- 質量収支および科学的妥当性の証明。
- [GPU_SURROGATE_ARCHITECTURE.md](GPU_SURROGATE_ARCHITECTURE.md): 複数環境を束ねるGPU FBA辞書、厳密フォールバック、RTX 4060受入試験。

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

---
**現在の状況**: 全主要開発フェーズを完了し、メンテナンスおよびコードベースの最適化フェーズにあります。
変更を加えた際は、必ず `pytest tests/test_project_integrity.py` を実行し、整合性を担保してください。
