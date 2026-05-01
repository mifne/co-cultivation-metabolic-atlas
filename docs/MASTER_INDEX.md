# プロジェクト・ドキュメント・マスターインデックス

本プロジェクト（天然ゴム分解・PHA変換コンソーシアム）の全記録を、整理・集約された以下の 3 つの大項目で管理しています。

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

---

## 📊 最終成果・証明資料
- **[FINAL_CONSORITUM_REPORT.md](FINAL_CONSORITUM_REPORT.md)**: 学習結果の要約、分解率・PHA収率の最終報告。
- **[FINAL_MASS_BALANCE_PROOF.png](FINAL_MASS_BALANCE_PROOF.png)**: 質量収支誤差 0.1% 未満を証明する決定的なグラフ。
- **[ULTIMATE_SYSTEM_PROOF.png](ULTIMATE_SYSTEM_PROOF.png)**: 代謝シナジーと AI 制御を多角的に可視化した最終証明図表。

---
**現在の状況**: 全主要開発フェーズを完了し、現在はメンテナンスおよびドキュメント整理フェーズにあります。
データログは `outputs/` ディレクトリに保存されています。
