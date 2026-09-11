# NS21 PHB/PHBVモデルの文献根拠と修正記録

## 結論

旧モデルの `PHB_syn` は、アセチルCoAからPHBまでを一反応に集約し、現行注釈でDUF3141タンパク質となる `A4W93_02445` に結び付けていた。この構造では、天然ゴムからNS21が生産すると報告されたPHBVの3HV成分、PhaB/PhaCの遺伝子ノックアウト、PhaZ分解を区別できない。そこで現行PGAP注釈とNS21の実験報告に合わせ、3HB枝と3HV枝を明示した。

## 根拠と実装

| 項目 | 文献・データベース上の根拠 | モデル上の処置 | 確度 |
|---|---|---|---|
| NS21ゲノム | 完全ゲノムCP015118.1、ゴム酸素添加酵素2遺伝子とβ酸化能が報告されている | RefSeq assembly `GCF_002116905.1` を参照 | 高 |
| PHA表現型 | 天然ゴムからPHBV、グルコースからPHBを生産し、窒素制限でPHBVが増加する | PHB（3HB）とPHBV中3HVを別poolとして追跡 | 高 |
| PhaC | 転写解析で `phaC` の関与が支持される。現行PGAPは `A4W93_RS10540` をclass I PHA synthaseと注釈 | モデルの旧locus tag `A4W93_10485` に対応させ、両重合反応へGPRを付与 | 高 |
| PhaB | 現行PGAPは隣接する `A4W93_RS10550` を `phbB` / acetoacetyl-CoA reductaseと注釈 | 旧tag `A4W93_10495` を3HB/3HV前駆体還元へ付与 | 高（3HB）／中（3HV基質範囲） |
| PhaA | 現行PGAPは `A4W93_RS10545` をacetyl-CoA C-acetyltransferaseと注釈 | 既存 `ACACT1r` の旧tag `A4W93_10490` を経路担当として注記 | 高 |
| PhaZ | PHA depolymerase破壊によるPHA増加が報告され、PGAPにも候補がある | 元素収支の取れた分解反応を追加するが、局在・速度未較正のためboundsを0に固定 | 構造は中、速度は未確定 |

主要資料：

- Tamamura et al. (2024), *Characterization of the conversion system of natural rubber to poly(3-hydroxyalkanoate) in Piscinibacter gummiphilus strain NS21T*, New Biotechnology 84, 77–85. https://doi.org/10.1016/j.nbt.2024.08.507
- Kasai et al. (2019), *Complete Genome Sequence of Rhizobacter gummiphilus NS21T*. https://doi.org/10.1128/MRA.00118-19
- NCBI RefSeq assembly GCF_002116905.1（PGAP 6.10, annotation release 2025-12-16）. https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_002116905.1/
- Chek et al. (2017), PhaCの構造、触媒残基と保存motif. https://doi.org/10.1038/s41598-017-05509-4

## 反応構造

1. PhaA相当：`2 acetyl-CoA ⇌ acetoacetyl-CoA + CoA`（既存 `ACACT1r`）
2. 3HB枝：`acetoacetyl-CoA + NADPH + H+ → (R)-3HB-CoA + NADP+`
3. 3HV枝：`3-oxopentanoyl-CoA + NADPH + H+ → (R)-3HV-CoA + NADP+`
4. PhaC：各hydroxyacyl-CoAからCoAを放出し、3HBまたは3HV repeat unitを細胞内poolへ移す
5. dFBA：両sinkのフラックスを個別積分し、総PHA、3HB量、3HV量、3HV mol%を保存する

GC/NMRで報告された基質依存表現型（天然ゴムではPHBV、グルコースではPHB）を守るため、既定値ではC30またはODTDが培地に存在するステップだけ3HV sinkを開く。将来、プロピオン酸等を直接供給する検証では `phv_requires_rubber_intermediate=False` とし、このgateを解除できる。

全反応は元素・電荷収支を満たす。`EX_pha_c` と `EX_phv_c` は名称上の互換性を保つためboundary reactionの形式だが、培地への分泌ではなく細胞内蓄積の動的会計にのみ使う。

## 断定していない事項

- 3HV前駆体に対するNS21 PhaB/PhaCの実効速度
- 天然ゴム条件での3HB/3HV比
- PhaZ候補ごとの細胞内局在、基質特異性、分解速度
- 最大PHA含有率と窒素制限に対する速度応答

これらはゲノム注釈だけでは決まらない。現在の `max_pha_fraction_g_gdcw=0.80` は数値暴走を防ぐ変更可能な安全上限であり、NS21の測定値ではない。学習結果を実験予測として用いる前に、乾燥菌体重量、PHA重量%、3HV mol%、NH4濃度、ゴム消費量の時系列で較正する。

## 既存GPU surrogateへの影響

反応・代謝物・目的関数の次元が変わったため、旧NS21モデルから作成した辞書とニューラルsurrogateは科学的に互換ではない。実行時の次元・署名検査で拒否し、修正版SBMLから候補辞書、QP投影行列、検証manifestを再生成する必要がある。旧成果物を自動的に正しいものとして流用してはならない。
