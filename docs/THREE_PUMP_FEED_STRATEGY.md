# 4ポンプ・pH制御付きジャーファーメンターの給排液設計

> 更新: サンプル採取を手動とし4本すべてを給餌へ使う最終案は
> [ONE_L_JAR_FERMENTER_CONTROL_DESIGN.md](ONE_L_JAR_FERMENTER_CONTROL_DESIGN.md)を参照する。

## 推奨結論

4台のポンプのうち3台を種別支援液、1台を培養前半の共通栄養液に使用し、共通栄養の
供給終了後に4台目を排液へ切り替える案は、**一方向に1回だけ切り替える運用なら採用候補**
になる。pHはジャーファーメンター内蔵制御に任せる。

推奨する初期割当ては次のとおりである。

| ポンプ | 増殖相（0–12 hの初期案） | PHA相 | 主目的 |
|---|---|---|---|
| P1 | OR16支援：コハク酸二Na | 必要時のみ低流量 | OR16立上げ、rubber/Lcp系の維持 |
| P2 | NS21支援：L-glutamate/MSG | 停止 | NS21立上げと窒素供給、後半のN制限 |
| P3 | LP支援：D-mannitol | pH・LP菌体を見てパルス | LP維持、BCAA等の授受を担う菌体確保 |
| P4 | 共通：yeast extract低濃度液 | 排液へ一方向切替 | 前半のAA・vitamin補完、後半の液量維持 |

「種特異的」は完全な選択性を意味しない。各炭素源は他菌にも利用され得るため、単独培養で
取り込みと増殖を確認し、最終的には「主対象菌に有利な支援液」として扱う。

## この割当てを選ぶ理由

- OR16は天然・合成ゴムを炭素・エネルギー源にでき、論文では10 mM succinateを含む
  W medium上でもrubber存在時にlcp転写が強く誘導された。現在のモデル操作である
  maltotrioseより、succinateの方が実験根拠を説明しやすい。
- NS21の報告培地はWx minimal salt medium＋10 mM glutamateである。glutamate feedを
  増殖相に限定し、PHA相で止めれば、NS21支援と窒素制限を1台で切り替えられる。
- L. plantarumはBCAA、arginine、glutamate、tryptophanなどへの依存が報告されるため、
  mannitolだけでは不十分である。前半のyeast extractを共通補完源とし、後半は停止する。
- PHAは一般に炭素過剰・栄養制限の二段階運転が用いられる。実モデルでも、pH-statと
  12時間後のyeast extract停止によりPHA 5.334 mmolを得た。

## モデルによるポンプ組合せ監査

現行GEMで `maltotriose / putrescine / mannitol / yeast-extract相当` の16組合せを総当たり
したところ、設定した高い供給レベルではyeast-extract相当だけで3種max–min LPが成立し、
共通増殖率は0.04263 h^-1だった。glucose、mannitol、glutamate、succinate、glycerolを
個別に加えても同じ律速に当たり、共通増殖率はほぼ変わらなかった。

これは「種別3液が不要」という意味ではない。現行のyeast-extract操作は実際のg/Lではなく、
仮想成分を固定比で直接mediumへ加える実装で、費用と濃度が校正されていない。種別液は
静的成立よりも、菌体比率、窒素切替、共通液削減を動的に制御するために使用する。

## P4を共通給餌から排液へ切り替える条件

推奨は、給餌と排液を何度も往復させず、増殖相からPHA相へ移るときに1回だけ切り替える
方式である。

1. 0–12 hまたはNH4が設定閾値へ下がるまではP4を共通栄養供給に使用する。
2. 共通栄養供給を停止し、feed reservoir側をクランプして物理的に切り離す。
3. 滅菌済みの別チューブ／別ポンプヘッド／selector valveを使い、P4駆動部を排液側へ接続する。
4. P1–P3の合計流量と同量をP4で排出し、液量または重量を一定にする。

同一チューブを培養液排出と栄養液供給の両方に使用してはならない。培養液が栄養リザーバーへ
逆流すると、feed全体が汚染される。ジャーが増殖相のfeed volumeを受け入れられない場合は、
P4切替案では前半の液量を維持できないため、別の低価格排液ポンプまたはsterile overflowが必要になる。

## 初期パイロット濃度

下表は1 Lの培養液へ最終的に到達させる初回スクリーニング濃度であり、確定レシピではない。

| 液 | 初期目標 | 1 L当たり試薬量 | 備考 |
|---|---:|---:|---|
| OR16/succinate | 5–10 mM | 無水二Na塩0.81–1.62 g | rubber誘導を確認しながら最小化 |
| NS21/glutamate | 5–10 mM | L-Glu 0.74–1.47 g相当 | 12 h以降停止しNH4/Gluを枯渇 |
| LP/mannitol | 5–10 mM | 0.91–1.82 g | pH低下を見てパルス化 |
| 共通/yeast extract | 1–2 g/L累積 | 1–2 g | 前半のみ、後半停止 |

無機塩、trace metals、phosphate、vitaminsのうち安定なものは、ポンプを消費せずinitial basal
mediumへ入れる。天然ゴムはジャーへ初期投入し、ポンプ輸送しない。

## 研究用試薬価格による概算

価格は2026年9月確認の税別または希望納入価格。送料、値引き、basal salts、rubber、pH titrantを
含まない。

| 原料 | 容量・価格 | 単価 | 10 mMまたは推奨量の概算/L |
|---|---:|---:|---:|
| コハク酸二Na（無水） | 500 g・6,660円 | 13.32円/g | 21.6円 |
| L-glutamic acid | 500 g・6,500円 | 13.0円/g | 19.1円 |
| D-mannitol | 500 g・3,200円 | 6.4円/g | 11.7円 |
| Yeast extract | 500 g・8,000円 | 16円/g | 16–32円 |
| 硫酸アンモニウム（比較） | 500 g・3,000円 | 6円/g | 低価格だが独立ポンプ不足 |
| Glycerol phosphate混合物 | 10 g・34,900円 | 3,490円/g | 高価・直接採用しない |
| Casamino acids | 500 g・21,500円 | 43円/g | YEより高くvitamin補完も必要 |

推奨上限の10 mM succinate、10 mM glutamate、10 mM mannitol、yeast extract 1–2 g/Lを
すべて使用しても、主要4原料は概算 **68–84円/L**、5 L仕込みで **約342–422円/run** である。
実際には5 mMから開始すればさらに下げられる。食品・発酵グレードへ移行する前に、まず
試薬グレードで再現性と純度影響を確認する。

## 比較した別案

### 機能別3液＋常時排液

P1を炭素、P2を窒素、P3をyeast extract、P4を常時排液とする案は、C/N比を最も直接的に
操作でき、RLにも扱いやすい。PHA最適化だけを優先するならこちらが第一候補である。

### 種別3液＋P4相切替

本提案は各菌の立上げと菌体比率を直接調整しやすい。P2 glutamateを止めることで窒素制限も
作れるため、3種共存を主要研究目的に含める場合はこちらを第一パイロット候補とする。
ただしP4の無菌切替と前半の液量増加を解決する必要がある。

## 次の実験ゲート

1. 各支援液を単独培養へ添加し、対象菌以外も利用するか測定する。
2. 1 L以下で5 mM、10 mMの2段階を比較する。
3. yeast extract 0、0.5、1、2 g/Lで3種維持と酸生成を測定する。
4. pH 6.2、6.5、7.0を比較し、塩基消費量も記録する。
5. NH4、glutamate、PHA、3種qPCR/CFUを測り、P4切替条件を時間固定から状態依存へ変更する。
6. この対照制御で168 hを通過してからRLの行動空間をP1–P4へ合わせる。

## 参考情報

- OR16 rubber/Lcp: https://pmc.ncbi.nlm.nih.gov/articles/PMC7413915/
- NS21 culture condition: https://www.tandfonline.com/doi/abs/10.1080/09168451.2016.1263147
- L. plantarum nutrient requirements: https://pmc.ncbi.nlm.nih.gov/articles/PMC1287688/
- L. plantarum MRS medium: https://mediadive.dsmz.de/medium/11?ccno=DSM+20174
- Two-stage PHA production: https://pmc.ncbi.nlm.nih.gov/articles/PMC5597195/
- Disodium succinate price: https://labchem-wako.fujifilm.com/jp/product/detail/W01W0104-2813.html
- L-glutamic acid price: https://labchem-wako.fujifilm.com/jp/product/detail/W01W0107-0050.html
- D-mannitol price: https://www.yone-yama.co.jp/shiyaku/search/shosai-04572.html
- Yeast extract price: https://labchem-wako.fujifilm.com/jp/product_data/docs/03694014_pamphlet.pdf
- Ammonium sulfate price: https://labchem-wako.fujifilm.com/jp/product/detail/W01W0101-0345.html
- Glycerol phosphate price: https://labchem-wako.fujifilm.com/jp/product/detail/W01TRCG601603.html
- Casamino acids price: https://labchem-wako.fujifilm.com/jp/product/detail/W01BIKA1404.html
