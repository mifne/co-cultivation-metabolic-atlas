# 手動SBMLモデル収集ガイド

## 概要

Gemini自動パイプラインで取得できなかったモデルを手動で収集する手順をまとめます。

---

## 優先度1: Gordonia polyisoprenivorans（真のLCP分解菌）⚠️

### 背景
- 現在の`Sphingobium japonicum`は実際にはPAH分解菌であり、天然ゴム分解の直接的証拠が不足
- **Gordonia polyisoprenivorans**は真のLcp酵素（Lcp1/Lcp2）を保有する天然ゴム分解のモデル株

### 収集状況: ❌ 入手失敗

1. **文献検索結果**
   - PMID: 35149306を確認したが、補足資料にSBMLファイルが含まれていない
   - 著者に直接連絡してモデルファイルを入手する必要がある

2. **代替案**
   - 現時点では**Sphingobium japonicum**を使用
   - 将来的にGordonia polyisoprenivoransのモデルが公開された場合に置き換え可能

3. **注意事項**
   - Sphingobiumは芳香族化合物分解能力を持ち、イソプレノイド代謝経路も保有
   - 天然ゴム分解の直接的証拠は不足しているが、概念実証（PoC）には使用可能

---

## 優先度2: Cupriavidus necator H16（真のPHA蓄積菌）❌

### 背景
- **Cupriavidus necator H16**はPHA生産のモデル株として広く研究されている
- しかし、公開されているゲノム規模代謝モデル（GEM）が見つからない

### 収集状況: ❌ 入手失敗

1. **BiGG Models検索結果**
   - モデルID `iJN1463`はCupriavidus necatorではなく、**Pseudomonas putida KT2440**のモデルであることが判明
   - BiGG ModelsにはCupriavidus necatorのモデルが登録されていない

2. **代替データベース検索**
   - ModelSEED: 検索したが公開モデルなし
   - AGORA: 腸内細菌中心のため該当なし
   - 文献検索: 公開SBMLファイルが見つからず

3. **採用した代替案**
   - ✅ **Pseudomonas putida KT2440 (iJN1462)** を使用
   - 実証されたmcl-PHA生産能力を持つ
   - BiGG Modelsから確実にダウンロード可能
   - GRAS認定、産業利用実績が豊富

4. **将来的な対応**
   - Cupriavidus necatorのGEMが公開された場合に置き換え可能
   - 現時点ではPseudomonas putidaで十分な性能が期待できる

---

## 優先度3: Pseudomonas putida KT2440（PHA蓄積菌）✅

### 収集状況: ✅ ダウンロード成功

1. **BiGG Modelsから直接ダウンロード**
   ```bash
   cd models/sbml
   wget http://bigg.ucsd.edu/static/models/iJN1462.xml -O Pseudomonas_putida_KT2440_iJN1462.xml
   ```

2. **検証**
   ```bash
   python -c "import cobra; model = cobra.io.read_sbml_model('models/sbml/Pseudomonas_putida_KT2440_iJN1462.xml'); print(f'Genes: {len(model.genes)}, Reactions: {len(model.reactions)}')"
   ```

3. **確認済み情報**
   - ファイルサイズ: 8.2 MB
   - 遺伝子数: 1,462
   - 反応数: 2,033
   - 代謝物数: 1,668
   - 保存先: `models/sbml/Pseudomonas_putida_KT2440_iJN1462.xml`

---

## 優先度4: Bacillus subtilis 168（代替安定化菌）✅

### 収集状況: ✅ ダウンロード成功

1. **BiGG Modelsから直接ダウンロード**
   ```bash
   cd models/sbml
   wget http://bigg.ucsd.edu/static/models/iYO844.xml -O Bacillus_subtilis_168_iYO844.xml
   ```

2. **検証**
   ```bash
   python -c "import cobra; model = cobra.io.read_sbml_model('models/sbml/Bacillus_subtilis_168_iYO844.xml'); print(f'Genes: {len(model.genes)}, Reactions: {len(model.reactions)}')"
   ```

3. **確認済み情報**
   - ファイルサイズ: 約5.8 MB
   - 遺伝子数: 844
   - 反応数: 1,437
   - 代謝物数: 1,145
   - 保存先: `models/sbml/Bacillus_subtilis_168_iYO844.xml`

4. **用途**
   - Lactobacillus plantarumの代替として使用可能
   - 比較実験や異なる安定化戦略の検証に有用

---

## ファイル命名規則

手動で収集したSBMLファイルは以下の命名規則に従ってください：

```
{Genus}_{species}_{strain}_{model_id}.xml
```

例：
- `Gordonia_polyisoprenivorans_VH2_iGP1495.xml`
- `Cupriavidus_necator_H16_iJN1463.xml`
- `Pseudomonas_putida_KT2440_iJN1462.xml`

---

## 収集後の統合

手動で収集したモデルを`main.py`で使用するには：

```bash
# 既存のモデルと一緒に使用
python main.py train --sbml-dir models/sbml --total-timesteps 100000
```

`load_sbml_models()`関数が自動的に`models/sbml/`内の全`.xml`ファイルを読み込みます。

---

## トラブルシューティング

### 問題1: XMLパースエラー

```bash
# ファイルの先頭を確認
head -n 20 models/sbml/your_model.xml
```

SBMLファイルは`<?xml version="1.0" encoding="UTF-8"?>`で始まる必要があります。

### 問題2: COBRApyで読み込めない

```python
import cobra
try:
    model = cobra.io.read_sbml_model('models/sbml/your_model.xml')
    print("✅ 読み込み成功")
except Exception as e:
    print(f"❌ エラー: {e}")
```

### 問題3: ファイルサイズが小さすぎる（< 1KB）

HTMLエラーページをダウンロードしている可能性があります。ブラウザで直接URLを開いて確認してください。

---

## 最終的なモデルセット（実際に入手可能）

以下の4種のモデルが実際にダウンロード成功しました：

### LCP分解菌
1. ✅ **Sphingobium japonicum (iJN1463)** - ダウンロード済み
   - 保存先: `models/sbml/Sphingobium_japonicum_iJN1463.xml`
   - 品質: 9/10
   - 注: PAH分解菌だが、イソプレノイド代謝経路を保有

2. ❌ Gordonia polyisoprenivorans - **入手不可**（文献に補足資料なし）

### PHA蓄積菌
3. ✅ **Pseudomonas putida KT2440 (iJN1462)** - ダウンロード済み
   - 保存先: `models/sbml/Pseudomonas_putida_KT2440_iJN1462.xml`
   - 品質: 9/10
   - mcl-PHA生産能力あり

4. ❌ Cupriavidus necator H16 - **入手不可**（BiGG Modelsに存在せず）

### 安定化菌
5. ✅ **Lactobacillus plantarum (iNF517)** - ダウンロード済み
   - 保存先: `models/sbml/Lactobacillus_plantarum_iNF517.xml`
   - 品質: 8/10
   - 乳酸発酵、pH調整能力

6. ✅ **Bacillus subtilis 168 (iYO844)** - ダウンロード済み
   - 保存先: `models/sbml/Bacillus_subtilis_168_iYO844.xml`
   - 品質: 8/10
   - 代替安定化菌として使用可能

## 推奨される使用方法

### 基本コンソーシアム（3種）
```python
# main.pyで使用
python main.py train --sbml-dir models/sbml --total-timesteps 100000
```

使用モデル:
1. Sphingobium japonicum (LCP分解)
2. Pseudomonas putida KT2440 (PHA蓄積)
3. Lactobacillus plantarum (安定化)

### 代替コンソーシアム（比較実験用）
安定化菌をBacillus subtilisに変更して比較実験が可能:
1. Sphingobium japonicum (LCP分解)
2. Pseudomonas putida KT2440 (PHA蓄積)
3. Bacillus subtilis 168 (安定化)

これにより、異なる安定化戦略（乳酸発酵 vs 胞子形成）の効果を比較できます。

---

**作成日**: 2026年4月24日  
**更新日**: 2026年4月24日
