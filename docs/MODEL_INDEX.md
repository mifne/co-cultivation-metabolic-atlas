# 代謝モデル・インデックス (MODEL_INDEX.md)

本プロジェクトで使用されている全ゲノムスケール代謝モデル（GSMM）の由来、改修履歴、および検証ステータスを管理します。

## 1. コンソーシアム構成モデル一覧

| 微生物名 | モデルファイルパス | 由来 (Base) | 役割 | ステータス |
| :--- | :--- | :--- | :--- | :--- |
| **Actinoplanes sp. OR16** | `models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml` | CarveMe (AP019371.1) | ゴム分解 (Lcp型) | 検証済み(速度パラメータは未校正、[THREE_SPECIES_COEXISTENCE_SUMMARY_20260911.docx](THREE_SPECIES_COEXISTENCE_SUMMARY_20260911.docx)第8章参照) |
| **Rhizobacter gummiphilus NS21** | `models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml` | CarveMe + manual curation (GCF_002116905.1) | ゴム分解 (Rox型) & PHB/PHBV変換 | 構造検証済み・速度未較正 |
| **Propionibacterium freudenreichii (Pf)** | `models/sbml/helper_candidates/Propionibacterium_freudenreichii_shermanii_curated.xml` | 詳細は同ディレクトリの `*_provenance.json` | 乳酸→プロピオン酸供給によるPHBV組成(3HV)支援 | 構造は使用中(`main.py`から読み込み)。酸素表現型・プロピオン酸生産速度は未校正 |

**現行の第3菌種は P. freudenreichii です。** 当初予定していた *Lactobacillus plantarum* (iNF517)
はメリットが乏しいと判断され不採用となりました。モデルファイル自体は
`models/sbml/final_consortium/Lactobacillus_plantarum.xml` として残っていますが、
`main.py` を含む現行コードはこれを読み込みません(下表参照)。

---

## 2. モデル別の詳細と改修履歴

### 🧬 Actinoplanes sp. OR16
- **ゲノム:** NCBI AP019371.1
- **改修スクリプト:** `scripts/modify_or16_model.py`
- **主要な変更点:**
    - **Lcp経路の実装**: 遺伝子 `G_ACTI_59630`(lcp1), `G_ACTI_59640`(lcp2),
      `G_ACTI_69520`(lcp3) を反応`R_LCP`(Lcp-catalysed oxidative endo-cleavage of
      natural rubber)へOR条件で紐付け。この3遺伝子はOR16のlcp遺伝子を直接特徴づけた
      文献(Gibu et al. 2020, *Appl Microbiol Biotechnol* 104:7367-7376,
      https://doi.org/10.1007/s00253-020-10700-1)のtBLASTn相同性検索結果
      (lcp1=ACTI_59630, lcp2=ACTI_59640, lcp3=ACTI_69520)と一致する。
      > **訂正記録(2026-09-12)**: 本節は従前`ACTI_28730`,`ACTI_28740`,`ACTI_37800`と
      > 記載していたが、これは誤りだった。実際のSBMLモデル
      > (models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml)を全反応にわたって
      > 監査した結果、`ACTI_28730`等はゲノムアノテーション上は実在する
      > (locus_tag/protein=hypothetical protein)ものの、`fbc:geneProduct`として
      > "lcp1/2/3"と命名されているだけで、**どの反応の`geneProductAssociation`にも
      > 使われていない孤立エントリ**だった。実際に`R_LCP`反応を駆動しているのは
      > `ACTI_59630`/`ACTI_59640`/`ACTI_69520`(Gibu et al. 2020と一致、遺伝子座番号も
      > lcp1・lcp2が隣接[オペロン]・lcp3が離れているという同論文の記述と整合)である。
      > **結論: モデルの反応ロジックは正しい文献に基づいて実装済み。本ドキュメントの
      > 記載が古かった/誤っていただけ。** `ACTI_28730`/`ACTI_28740`/`ACTI_37800`の
      > 孤立した`geneProduct`エントリをSBMLから削除するかは別途判断すること。
    - **反応トポロジー**: 天然ゴム (`rubber_e`) をオリゴマー (`rubber_fragment_e`) へエンド型切断し、同時に資化する経路を構築。
    - **生化学的補正 (OxiAB)**: ゴム切断後のアルデヒドを酸化する OxiAB ホモログ (`ISOP_ALDH`) の電子受容体を、誤った $NAD^+$ から**科学的に正しいシトクロムc (`ficytc_c`) へ修正**。これにより呼吸鎖と連動し、致死的なレドックスアンバランス（過剰還元の蓄積）を解消。
    - **質量・電子バランス補正**: ベータ酸化を一括で表現する `ISOP_ACS` 反応において、CoAの消費不足による質量保存の違反（FBAによるペナルティ）を修正し、$7 NADH$ / $7 FADH_2$ の生成を厳密に定義。
    - **境界条件の開放**: `EX_rubber_bulk_e` の LowerBound を -1000 に設定し、シミュレーター側での動的制御を可能にした。
    - **ギャップフィリング**: M9培地での増殖を可能にするため、必須アミノ酸合成経路を補完。
- **検証結果:** ゴム単一炭素源での増殖速度 0.0804 を達成。修正により、最大比ゴム取り込み速度 ($q_{rubber\_max}$) は `4.30 g/gDW/h` に到達（ポテンシャルの解放）。

### 🧬 Rhizobacter gummiphilus NS21
- **ゲノム:** RefSeq GCF_002116905.1 / GenBank CP015118.1
- **改修スクリプト:** `scripts/model_ops/curate_ns21_pha_pathway.py`
- **主要な変更点:**
    - **Rox経路 (LatA1/A2) の実装**: OR16が放出した `rubber_fragment_e` を資化し、精密中間体 `ODTD` へと変換するエキソ型切断を実装。
    - **PHA合成経路**: 現行PGAP注釈に基づき `A4W93_10485`（現行別名 `A4W93_RS10540`; `phaC`）と `A4W93_10495`（`A4W93_RS10550`; `phbB`）を採用。誤ってPhaCに割り当てられていた `A4W93_02445` を除外した。
    - **PHBV表現**: 3HBと3HVの前駆体還元、PhaC重合、細胞内蓄積sinkを別反応として実装し、3HVモル分率をdFBAで追跡可能にした。
    - **PHA分解**: PhaZ候補反応は記載したが、速度未較正のためFBAでは無効化した。
    - **PHA蓄積パスの導通**: `EX_pha_c` と `EX_phv_c` は分泌ではなく細胞内動的蓄積sinkとして扱う。
    - **代謝トポロジー**: ODTD -> アセチルCoA -> PHA の流路を最適化。
- **検証結果:** PhaB/PhaC/PhaZ反応の元素・電荷収支、PhaB/PhaCノックアウトによる両枝遮断、旧・現行locus tag対応を自動試験する。絶対生産速度と最大含有率は実験較正前であり、予測値として確定しない。

### 🧬 Propionibacterium freudenreichii (Pf) — 現行の第3菌種
- **モデルファイル:** `models/sbml/helper_candidates/Propionibacterium_freudenreichii_shermanii_curated.xml`
- **役割:** 乳酸をプロピオン酸へ変換し、NS21のPHBV合成に3HV前駆体を系内供給する。
- **現状の知見:** 3種優位は酸素供給条件(kLa)に対して非単調に変化し、条件依存的である
  (kLa中間域で最大+25.4%有利、kLa 2一定で-51.6%不利)。すべて計算(dFBA)上の結果であり、
  湿式実験による検証は未実施。詳細は
  [OXYGEN_SCHEDULE_THIRD_SPECIES_20260910.md](OXYGEN_SCHEDULE_THIRD_SPECIES_20260910.md)、
  [THREE_SPECIES_COEXISTENCE_SUMMARY_20260911.docx](THREE_SPECIES_COEXISTENCE_SUMMARY_20260911.docx)、
  校正すべきパラメータと実験手順は
  [CALIBRATION_EXPERIMENT_PROTOCOL_20260911.docx](CALIBRATION_EXPERIMENT_PROTOCOL_20260911.docx)を参照。
- **未校正・未実装の主な点:** Pfの酸素表現型(モデル上は閉鎖=嫌気固定)、B12授受経路
  (OR16/NS21側の受け取り経路は未実装)、プロピオン酸生産速度。

### 🧬 Lactobacillus plantarum (iNF517) — 不採用・記録として保持
- **由来:** 高品質な既存モデル iNF517 を採用。
- **当初の役割:** 他の 2 種が要求する微量栄養素（ビタミン等）の供給源、およびバイオサーファクタント（BS）の生産。
- **改修内容(当時):**
    - コンパートメントIDの標準化。
    - **BS分泌能力の導通**: `EX_biosurfactant_e` の UpperBound を開放し、界面活性剤による分解ブースト機能を有効化。
- **不採用の理由:** 計算研究でP. freudenreichiiと比較した結果、明確なメリットが
  確認できなかったため、第3菌種としては不採用となった。モデルファイルは
  `models/sbml/final_consortium/Lactobacillus_plantarum.xml` として残置しているが、
  現行コードからは読み込まれない。

---

## 3. モデル改修ワークフロー

新しい改修が必要な場合は、以下の手順に従ってください：
1. `scripts/modify_*.py` を修正して、意図した生化学反応を定義する。
2. スクリプトを実行し、`models/sbml/final_consortium/` 下の XML を更新する。
3. `tests/test_science_integrity.py` を実行して、科学的な整合性が保たれているか確認する。
