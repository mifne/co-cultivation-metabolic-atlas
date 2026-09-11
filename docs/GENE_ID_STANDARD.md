# コンソーシアムGEMのGene ID標準

> **2026-09-11時点の注記**: 本書のWCFS1に関する記述(Gene ID正規化の実施記録、
> GeneProduct件数等)は *L. plantarum* (WCFS1) を第3菌種としていた当時の記録です。
> 現行の第3菌種は *P. freudenreichii* (Pf) に変更されています
> ([PROJECT_MANAGEMENT.md](PROJECT_MANAGEMENT.md) Phase D参照)。Gene ID標準の
> 方針自体はPfにも適用できますが、Pf向けの正規化作業はまだ実施・記録されていません。

## 結論

Gene IDの不一致はGEMで広く報告されている相互運用性の問題であり、単一の万能Gene IDへ置換する標準は存在しない。本プロジェクトでは、既存のSBML-FBC、MIRIAM/Identifiers.org、NCBIおよびBiGG Modelsの方針に合わせ、次の二層方式を採用する。

1. 各GEMのGPRでは、対象ゲノム注釈の`locus_tag`を使用する。
2. プロジェクト横断処理では、`CCG_<organism_key>_<locus_tag>`を一意キーとする。
3. NCBI protein accession、旧SBML ID、FASTA ID、gene symbolは別名表に保持する。
4. 別名の解決には必ずorganism keyを要求する。gene symbolは重複するため自動解決に使用しない。
5. 根拠がないIDは推測せず、`unresolved`または`retired_unresolved`として隔離する。

## 既存標準との関係

- SBML Level 3 FBC v2は、反応とGeneProductを論理式で結ぶ`GeneProductAssociation`を定義する。ただし、外部データベース間で共通となるGene ID文字列自体は規定しない。
- MIRIAM/Identifiers.orgは、名前空間付きURIによってモデル要素を外部レコードへ対応付ける。IDを一つに潰すのではなく、明示的なcross-referenceを保持する考え方である。
- NCBIは原核生物の全Geneに、ゲノム内で一意なsystematic identifierとして`locus_tag`を要求する。
- BiGG ModelsもGene IDに対象ゲノム注釈のlocus IDを採用し、モデル固有Geneと注釈固有Geneを分離している。
- MEMOTEはGPRおよび注釈の完全性・一貫性を品質管理対象としている。

したがって、独自IDだけに置換するのではなく、正式なlocus tagをGPRの主IDとし、独自IDは株間衝突を防ぐレジストリキーとして使う。

## 実施した正規化

| モデル | 正規化前の主表記 | 本番GPRの表記 | 結果 |
|---|---|---|---|
| OR16 | `lcl_AP019371_1_prot_BBH...` | `ACTI_*` | 1,589件のFASTA由来IDをlocus tagへ変換 |
| NS21 | `lcl_CP015118_1_prot_ARN...`と`A4W93_*`の混在 | `A4W93_*` | 1,389件を変換し、同一遺伝子2組を統合 |
| WCFS1 | `lp_*`、`GntK`、`lp1406`等の混在 | `lp_*`またはplasmid locus tag | `GntK`を`lp_1250`へ統合し、冗長GPRを論理簡約 |

WCFS1の`GNK`は`GntK or lp_1250`から`lp_1250`へ簡約した。`DAPRPL`の`(lp_2019 or lp1406) and lp_2019`はブール代数上`lp_2019`と厳密に等価であるため簡約した。`lp1406`を`lp_1406`と推測して置換してはいない。旧IDは`retired_unresolved`としてレジストリに保存している。

正規化後の本番モデルには、OR16 1,596、NS21 1,392、WCFS1 913 GeneProductがある。このうち生物学的Geneは3,899件、`spontaneous` sentinelは2件である。レジストリは履歴上の未解決ID 1件を含む3,902レコードと17,355件の別名を保持する。NS21の増分は、現行PGAP注釈で確認したPhaC/PhaB/PhaZ候補を旧locus tagへ対応付けた結果である。

## ファイル

- `config/gene_id_registry.json`: 株、assembly、locus tag規則、手動根拠
- `models/gene_registry/gene_id_registry.tsv`: 一遺伝子一行の主台帳
- `models/gene_registry/gene_id_aliases.tsv`: 旧ID・FASTA ID・protein accessionの別名表
- `models/gene_registry/gene_id_registry.json`: 監査集計
- `src/gene_id_registry.py`: 正規化、解決、検証API
- `scripts/model_ops/build_gene_id_registry.py`: 台帳生成とSBML正規化

再生成コマンド：

```bash
python3 scripts/model_ops/build_gene_id_registry.py --canonicalize-models
```

Pythonから旧IDを解決する例：

```python
from src.gene_id_registry import resolve_gene_id

resolve_gene_id("NS21", "ARN18758.1")
# CCG_NS21_A4W93_01825
```

`select_consortium_models()`は、旧形式のGene IDが本番モデルへ再混入した場合に例外を出す。これにより、表記揺れがノックアウト対象やGPR検索を静かに外すことを防止する。

## 今後の運用規則

1. GEMを更新するときは、必ず同じassembly accessionのFAA/GBFFを保存する。
2. Gene追加時はgene symbolではなくlocus tagをGPRへ設定する。
3. protein accessionやUniProt IDは主IDではなくcross-referenceとして追加する。
4. assembly更新時は旧・新注釈を配列対応付けし、別名表へ移行履歴を追加する。
5. KO/eggNOG/ECは機能・オルソログ分類であり、株内Geneの一意IDとして使用しない。
6. ノックアウト入力は`organism_key + alias`で解決し、解決不能または多義的なgene symbolを拒否する。

## 限界

- locus tagはゲノム注釈に依存するため、assemblyまたは注釈版が変わると再対応付けが必要になる。
- WCFS1モデルにはFASTA/GBFFに基づく全Geneのprotein accession対応がまだない。現時点ではassembly accessionとlocus tagを主根拠とする。
- Gene IDの統一はGPRの同一性を改善するが、反応アノテーションやオルソログ機能の正しさを保証しない。

## 参考文献・仕様

- Juty et al. (2012), Identifiers.org and MIRIAM Registry, *Nucleic Acids Research* 40:D580–D586. https://academic.oup.com/nar/article/40/D1/D580/2903100
- King et al. (2016), BiGG Models, *Nucleic Acids Research* 44:D515–D522. https://pmc.ncbi.nlm.nih.gov/articles/PMC4702785/
- Lieven et al. (2020), MEMOTE, *Nature Biotechnology* 38:272–276. https://www.nature.com/articles/s41587-020-0446-y
- SBML Level 3 Flux Balance Constraints version 2 specification. https://sbml.org/documents/specifications/level-3/version-1/fbc/
- NCBI Prokaryotic Genome Annotation Guide. https://www.ncbi.nlm.nih.gov/genbank/genomesubmit_annotation/
