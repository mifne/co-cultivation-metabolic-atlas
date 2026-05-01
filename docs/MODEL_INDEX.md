# 代謝モデル・インデックス (MODEL_INDEX.md)

本プロジェクトで使用されている全ゲノムスケール代謝モデル（GSMM）の由来、改修履歴、および検証ステータスを管理します。

## 1. コンソーシアム構成モデル一覧

| 微生物名 | モデルファイルパス | 由来 (Base) | 役割 | ステータス |
| :--- | :--- | :--- | :--- | :--- |
| **Actinoplanes sp. OR16** | `models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml` | CarveMe (AP019371.1) | ゴム分解 (Lcp型) | 検証済み |
| **Rhizobacter gummiphilus NS21** | `models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml` | CarveMe (GCF_001511185.1) | ゴム分解 (Rox型) & PHA変換 | 検証済み |
| **Lactobacillus plantarum** | `models/sbml/final_consortium/Lactobacillus_plantarum.xml` | iNF517 (Validated) | 代謝安定化 (Buffer) | 検証済み |

---

## 2. モデル別の詳細と改修履歴

### 🧬 Actinoplanes sp. OR16
- **ゲノム:** NCBI AP019371.1
- **改修スクリプト:** `scripts/modify_or16_model.py`
- **主要な変更点:**
    - **Lcp経路の実装**: 遺伝子 `ACTI_28730`, `ACTI_28740`, `ACTI_37800` を紐付け。
    - **反応トポロジー**: 天然ゴム (`rubber_e`) をオリゴマー (`rubber_fragment_e`) へエンド型切断し、同時に資化する経路を構築。
    - **境界条件の開放**: `EX_rubber_e` の LowerBound を -1000 に設定し、シミュレーター側での動的制御を可能にした。
    - **ギャップフィリング**: M9培地での増殖を可能にするため、必須アミノ酸合成経路を補完。
- **検証結果:** ゴム単一炭素源での増殖速度 0.0804 を達成。

### 🧬 Rhizobacter gummiphilus NS21
- **ゲノム:** RefSeq GCF_001511185.1
- **改修スクリプト:** `scripts/modify_ns21_model.py`
- **主要な変更点:**
    - **Rox経路 (LatA1/A2) の実装**: OR16が放出した `rubber_fragment_e` を資化し、精密中間体 `ODTD` へと変換するエキソ型切断を実装。
    - **PHA合成経路**: アセチルCoAから PHA (PHB) を蓄積する `phaCAB` 経路 (`PHB_syn`) をキュレーション。
    - **PHA排出パスの導通**: `EX_pha_c` の UpperBound を 1000 に設定し、蓄積フラックスが FBA 計算上流れるように修正。
    - **代謝トポロジー**: ODTD -> アセチルCoA -> PHA の流路を最適化。
- **検証結果:** ゴムオリゴマーからの PHA 蓄積を確認 (Max Production 500.0)。

### 🧬 Lactobacillus plantarum (iNF517)
- **由来:** 高品質な既存モデル iNF517 を採用。
- **役割:** 他の 2 種が要求する微量栄養素（ビタミン等）の供給源、およびバイオサーファクタント（BS）の生産。
- **改修内容:** 
    - コンパートメントIDの標準化。
    - **BS分泌能力の導通**: `EX_biosurfactant_e` の UpperBound を開放し、界面活性剤による分解ブースト機能を有効化。

---

## 3. モデル改修ワークフロー

新しい改修が必要な場合は、以下の手順に従ってください：
1. `scripts/modify_*.py` を修正して、意図した生化学反応を定義する。
2. スクリプトを実行し、`models/sbml/final_consortium/` 下の XML を更新する。
3. `tests/test_science_integrity.py` を実行して、科学的な整合性が保たれているか確認する。
