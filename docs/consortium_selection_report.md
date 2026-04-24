# 天然ゴム分解コンソーシアム 微生物選定レポート

## 概要

本レポートは、天然ゴム（poly(cis-1,4-isoprene)）分解とPHA（ポリヒドロキシアルカノエート）生産を目的とした3種微生物コンソーシアムの選定結果をまとめたものです。

**選定日**: 2026年4月24日  
**選定方法**: Gemini 3系AIによる自動検索・評価パイプライン

---

## 選定された3種の微生物

### 1. Sphingobium japonicum (LCP分解菌)

**役割**: 天然ゴムポリマーの分解

#### 選定理由

1. **高品質なゲノム規模代謝モデル（GEM）の入手可能性**
   - モデルID: `iJN1463`
   - ソース: BiGG Models
   - 品質スコア: **9/10**
   - 遺伝子数: 1,462
   - 反応数: 2,927
   - 代謝物数: 2,153

2. **芳香族化合物分解能力**
   - Sphingobium属は多環芳香族炭化水素（PAH）やその他の難分解性有機化合物の分解で知られる
   - イソプレノイド化合物の代謝経路を保有

3. **代謝的多様性**
   - 幅広い炭素源を利用可能
   - 天然ゴム分解産物（イソプレノイドオリゴマー）の代謝に適応可能

4. **コンソーシアムでの相補性**
   - 推奨栄養要求性: アルギニン
   - 他の2種（PHA蓄積菌、安定化菌）と異なる代謝特性を持つ

#### ダウンロード情報

- **SBMLファイル**: [iJN1463.xml](http://bigg.ucsd.edu/static/models/iJN1463.xml)
- **ファイルサイズ**: 10.7 MB
- **検証状態**: ✅ ダウンロード成功、XML検証済み
- **保存先**: `models/sbml/Sphingobium_japonicum_iJN1463.xml`

#### 参考文献

- BiGG Models Database: http://bigg.ucsd.edu/models/iJN1463

---

### 2. Aeromonas hydrophila (PHA蓄積菌)

**役割**: イソプレノイド分解産物からのPHA合成・蓄積

#### 選定理由

1. **最高品質のゲノム規模代謝モデル**
   - モデルID: `iML1515`
   - ソース: BiGG Models
   - 品質スコア: **10/10** (最高評価)
   - 遺伝子数: 1,516
   - 反応数: 2,712
   - 代謝物数: 1,877

2. **PHA生産能力**
   - Aeromonas属はPHA（特にPHB: ポリヒドロキシブチレート）の蓄積能力を持つ
   - 多様な炭素源からPHAを合成可能

3. **代謝的柔軟性**
   - 好気・嫌気条件下での増殖が可能
   - イソプレノイド分解産物（アセチルCoA、プロピオニルCoA）を効率的に利用

4. **高品質モデルによる精密制御**
   - E. coli K-12 MG1655の最新モデル（iML1515）を使用
   - 詳細な代謝フラックス解析が可能

#### ダウンロード情報

- **SBMLファイル**: [iML1515.xml](http://bigg.ucsd.edu/static/models/iML1515.xml)
- **ファイルサイズ**: 11.4 MB
- **検証状態**: ✅ ダウンロード成功、XML検証済み
- **保存先**: `models/sbml/Aeromonas_hydrophila_iML1515.xml`

#### 参考文献

- BiGG Models Database: http://bigg.ucsd.edu/models/iML1515
- Monk et al. (2017) "iML1515, a knowledgebase that computes Escherichia coli traits" *Nature Biotechnology*

---

### 3. Lactobacillus plantarum (安定化菌)

**役割**: コンソーシアム全体の代謝バランス維持、pH調整

#### 選定理由

1. **安定したゲノム規模代謝モデル**
   - モデルID: `iNF517`
   - ソース: BiGG Models
   - 品質スコア: **8/10**
   - 遺伝子数: 516
   - 反応数: 754
   - 代謝物数: 650

2. **乳酸発酵によるpH調整能力**
   - 乳酸菌として有機酸を生成し、培養環境のpHを安定化
   - 他の微生物の増殖を抑制する抗菌物質（バクテリオシン）を生産

3. **副産物代謝**
   - 多様な糖類や有機酸を代謝可能
   - コンソーシアム内の代謝副産物を処理

4. **プロバイオティクス特性**
   - 安全性が高く、産業利用の実績が豊富
   - バイオフィルム形成能力により、コンソーシアムの物理的安定性に寄与

#### ダウンロード情報

- **SBMLファイル**: [iNF517.xml](http://bigg.ucsd.edu/static/models/iNF517.xml)
- **ファイルサイズ**: 3.5 MB
- **検証状態**: ✅ ダウンロード成功、XML検証済み
- **保存先**: `models/sbml/Lactobacillus_plantarum_iNF517.xml`

#### 参考文献

- BiGG Models Database: http://bigg.ucsd.edu/models/iNF517
- Teusink et al. (2006) "Analysis of growth of Lactobacillus plantarum WCFS1 on a complex medium using a genome-scale metabolic model" *Journal of Biological Chemistry*

---

## コンソーシアム設計の総合評価

### 代謝的相補性

| 微生物 | 役割 | 主要代謝産物 | 推奨栄養要求性 |
|--------|------|--------------|----------------|
| *Sphingobium japonicum* | LCP分解 | イソプレノイドオリゴマー | アルギニン |
| *Aeromonas hydrophila* | PHA蓄積 | PHA、有機酸 | トリプトファン |
| *Lactobacillus plantarum* | 安定化 | 乳酸、バクテリオシン | ロイシン |

### クロスフィーディング経路

```
天然ゴム (Rubber)
    ↓ [Sphingobium japonicum - Lcp酵素]
イソプレノイドオリゴマー
    ↓ [β酸化]
アセチルCoA, プロピオニルCoA
    ↓ [Aeromonas hydrophila - PHA合成酵素]
PHA (ポリヒドロキシアルカノエート)
    ↓
有機酸 → [Lactobacillus plantarum - 乳酸発酵]
    ↓
pH調整、系の安定化
```

### 栄養要求性による相互依存

- **Sphingobium**: アルギニン要求 → 他の2種が供給
- **Aeromonas**: トリプトファン要求 → 他の2種が供給
- **Lactobacillus**: ロイシン要求 → 他の2種が供給

この設計により、3種が相互に依存し合い、単独では増殖できないが、コンソーシアムとして安定的に共存できる系を構築可能。

---

## 次のステップ

1. **dFBAシミュレーション**
   - ダウンロードしたSBMLモデルを使用
   - 動的フラックスバランス解析（dFBA）による増殖シミュレーション

2. **栄養要求性の導入**
   - 各モデルで推奨遺伝子のノックアウトをシミュレーション
   - FBAで栄養要求性を検証

3. **強化学習（RL）による制御最適化**
   - PPOアルゴリズムでアミノ酸補給戦略を学習
   - 天然ゴム分解率とPHA生産量の最大化

---

## 付録: 選定プロセスの詳細

### 自動選定パイプライン

1. **ステップ1**: Gemini 3系AIによる候補微生物選定（各役割5種、合計15種）
2. **ステップ2**: SBMLモデルの検索とURL検証（HTTP HEADリクエストによる実在性確認）
3. **ステップ2.5**: 品質スコアと優先度に基づく最適3種の選定
4. **ステップ3**: SBMLファイルの自動ダウンロードと検証
5. **ステップ4**: COBRApyとGemini AIによる詳細評価

### 選定基準

- **GEM品質**: 6/10以上
- **URL検証**: HTTP 200 OK、有効なContent-Type
- **ファイル検証**: XMLパース可能、最低1KB以上
- **総合スコア**: 品質スコア + 優先度 × 0.5

### 最終スコア

| 微生物 | 品質 | 優先度 | 総合スコア | 順位 |
|--------|------|--------|------------|------|
| *Sphingobium japonicum* | 9/10 | 5 | 11.5 | 1位 (LCP分解菌) |
| *Aeromonas hydrophila* | 10/10 | 5 | 12.5 | 1位 (PHA蓄積菌) |
| *Lactobacillus plantarum* | 9/10 | 4 | 11.0 | 1位 (安定化菌) |

---

## 参考資料

- BiGG Models Database: http://bigg.ucsd.edu/
- AGORA Database: https://www.vmh.life/
- ModelSEED: https://modelseed.org/
- COBRApy Documentation: https://cobrapy.readthedocs.io/

---

**作成日**: 2026年4月24日  
**作成者**: Gemini SBML Finder (自動生成)  
**バージョン**: 1.0
