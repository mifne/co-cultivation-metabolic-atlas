# 三種共培養・流加液候補の実験検証計画

## 目的と判定対象

計算で選定した **L-glutamate + L-isoleucine + pyridoxamine** が、Actinoplanes sp. OR16、Rhizobacter gummiphilus NS21、Lactiplantibacillus plantarum WCFS1の三種共存、ゴム分解、PHA蓄積を実際に支えるかを検証する。

FBAが示せるのは、モデル化された反応・交換境界・完全混合条件での化学量論的実行可能性である。次の値は実験なしには確定できない。

| 未確定データ | なぜ計算だけでは決まらないか | 主な測定法 |
|---|---|---|
| 各菌の絶対菌数と生残 | 混合培養のODは菌種を区別できない | 種特異的qPCR/ddPCR、必要に応じCFU |
| 実増殖速度・競争係数 | FBA目的関数と実際の増殖制御は一致しない | 時系列の絶対菌数、単独・二種・三種培養比較 |
| Glu、Ile、B6 vitamerの取り込み・分泌 | 交換境界は主にモデル仮定である | LC-MS/HPLCによる培地上清の時系列定量 |
| Pyridoxamineとpyridoxineの代替性 | 輸送体、vitamer変換、安定性が未校正 | 等モル置換試験とB6 vitamer定量 |
| ゴム切断速度と酸素依存性 | Lcp/Rox反応はGEM内で粗視化されている | 残存ゴム乾燥重量、FTIR、SEC/GPC、生成物HPLC/LC-MS |
| OR16のLcp1/2/3、NS21のRoxA/B活性 | 遺伝子存在だけでは発現・酵素活性を保証しない | RT-qPCR、酸素消費、必要に応じ酵素活性試験 |
| PHA量・モノマー組成 | GEMの蓄積フラックスは細胞内ポリマー量そのものではない | 乾燥菌体、酸メタノリシス後GC-FID/GC-MS |
| pH、DO、kLa、撹拌・温度の影響 | 物質移動と局所濃度は完全混合仮定から外れる | オンラインpH/DO、撹拌・温度・酸/塩基添加量ログ |
| 流加液の安定性・ポンプ精度 | 低濃度B6の分解、吸着、脈動はモデル外 | 保存安定性試験、流量の重量校正、無菌試験 |

## 段階1: 小容量の濃度・欠損スクリーニング

まず単独培養と三種混合培養を小容量で行い、次の条件を各3生物学的反復でランダム配置する。

1. 基礎培地のみ（無流加対照）
2. 選定3成分の0.5×、1×、2×、5×
3. 2×配合からIleを除く
4. 2×配合からpyridoxamineを除く
5. 2×配合からGluを除く
6. pyridoxamineをpyridoxineへ等モル置換
7. Defined-10基準液
8. WCFS1単独では既知の増殖培地を陽性対照とする

0、4、8、12、24時間を基本採取点とし、総ODに加えて各菌の絶対量を種特異的qPCRで測定する。混合培養での菌種別定量にはqPCRが実際に用いられており、基質、代謝産物、pH制御と合わせて評価する設計が妥当である。プライマーはgene registryの各菌固有・単一コピー候補から選び、標準曲線、増幅効率、融解曲線、交差増幅なしを事前確認する。

## 段階2: 1 Lジャーファーメンターでの確認

小容量試験で有効濃度を確定してから、無流加、選定3成分、Defined-10の3条件を優先して各3生物学的反復で比較する。最初のモデル等価条件は、1 L槽に1本の流加ポンプを2 mL h⁻¹で用いる場合、Glu 0.03685 g L⁻¹、Ile 0.01413 g L⁻¹、pyridoxamine 0.000843 g L⁻¹を同一ストックに含める計算となる。ただし極めて希薄なので、段階1で得た有効倍率へ補正し、濃縮マスターストックから調製する。

- pHは装置の酸・塩基ポンプで一定化し、添加量を記録する。
- DO、撹拌速度、温度、流加実重量を連続記録する。
- サンプリングは手動で0、4、8、12、24時間に行い、採取体積を物質収支に反映する。
- qPCR、培地成分、主要有機酸、残存ゴム、ゴム切断生成物、PHAを同じ時点で対応づける。
- ゴムはロット、粒径、初期乾燥重量を揃え、無菌ゴム対照と菌体なし化学対照を置く。

## ゴム分解とPHAの測定

OR16では3つのLcp遺伝子が同定され、天然ゴム条件で誘導されること、NS21ではRoxA/RoxBが相乗的に働き、RoxBがC20以上のオリゴイソプレノイド、RoxAが主にODTD生成へ関与することが報告されている。したがって、ゴム分解は重量減少だけでなく、FTIRのカルボニル形成、SEC/GPCの分子量低下、ODTD/オリゴマー生成を組み合わせて判定する。

PHAは総菌体増加と分け、乾燥菌体重量に対するPHA含有率（% DCW）と培養液当たりPHA量（g L⁻¹）を併記する。酸メタノリシス後のGC-FID/GC-MSでモノマー組成を確認する。溶媒・濃酸・加熱を伴うため、ここでは概略に留め、所属機関の承認済みSOP、ドラフト、廃液手順に従う。

## 事前に固定する判定基準

- 24時間後に三菌すべての絶対量が初期値を下回らず、少なくとも一つ前の時点から正の純増を示す。
- 選定3成分がDefined-10に対して、事前に定めた非劣性幅内で三菌の共存を維持する。
- 欠損対照で予測されたIle・pyridoxamine依存性が再現するかを判定する。
- ゴム分解、PHA、菌体量を別々の主要評価項目として扱い、どれか一つだけで成功としない。
- パイロットは各3生物学的反復とし、モデル条件の反復計算を生物学的nとして数えない。
- 時系列は条件×時間の混合効果モデルまたは事前に定めた反復測定解析を用い、生データと除外理由を保存する。

## 参考文献

- Hébert et al. *Improvement of a chemically defined medium for the sustained growth of Lactobacillus plantarum: nutritional requirements*. https://pubmed.ncbi.nlm.nih.gov/17503149/
- Medina et al. *Prebiotics Mediate Microbial Interactions in a Consortium of the Infant Gut Microbiome*. https://pmc.ncbi.nlm.nih.gov/articles/PMC5666777/
- Yikmis et al. *Characterization of the genes responsible for rubber degradation in Actinoplanes sp. strain OR16*. https://pmc.ncbi.nlm.nih.gov/articles/PMC7413915/
- Birke et al. *Rhizobacter gummiphilus NS21 has two rubber oxygenases (RoxA and RoxB) acting synergistically in rubber utilisation*. https://pubmed.ncbi.nlm.nih.gov/30215127/
- Birke & Jendrossek. *Rubber oxygenases*. https://pmc.ncbi.nlm.nih.gov/articles/PMC6311187/
- Liu et al. *Enhancement of polyhydroxyalkanoate production by co-feeding lignin derivatives with glycerol in Pseudomonas putida KT2440*. https://pmc.ncbi.nlm.nih.gov/articles/PMC7792162/
