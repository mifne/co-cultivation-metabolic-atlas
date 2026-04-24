# 手動SBMLモデル収集ガイド

## 概要

Gemini自動パイプラインで取得できなかったモデルを手動で収集する手順をまとめます。

---

## 優先度1: Gordonia polyisoprenivorans（真のLCP分解菌）

### 背景
- 現在の`Sphingobium japonicum`は実際にはPAH分解菌であり、天然ゴム分解の直接的証拠が不足
- **Gordonia polyisoprenivorans**は真のLcp酵素（Lcp1/Lcp2）を保有する天然ゴム分解のモデル株

### 収集方法

1. **文献検索**
   - PMID: 35149306
   - タイトル: "A genome-scale metabolic model of the polyisoprene-degrading bacterium Gordonia polyisoprenivorans VH2"
   - URL: https://pubmed.ncbi.nlm.nih.gov/35149306/

2. **補足資料の確認**
   - 論文ページから "Supplementary Materials" を探す
   - SBMLファイル（通常 `.xml` または `.sbml` 形式）をダウンロード

3. **保存先**
   ```bash
   models/sbml/Gordonia_polyisoprenivorans_VH2.xml
   ```

4. **検証**
   ```bash
   python -c "import cobra; model = cobra.io.read_sbml_model('models/sbml/Gordonia_polyisoprenivorans_VH2.xml'); print(f'Genes: {len(model.genes)}, Reactions: {len(model.reactions)}')"
   ```

---

## 優先度2: Cupriavidus necator H16（真のPHA蓄積菌）

### 背景
- 現在の`Aeromonas hydrophila`は実際にはE. coli K-12のモデル（iML1515）を使用
- **Cupriavidus necator H16**はPHA生産のモデル株として広く研究されている

### 収集方法

1. **BiGG Modelsから直接ダウンロード**
   ```bash
   cd models/sbml
   wget http://bigg.ucsd.edu/static/models/iJN1463.xml -O Cupriavidus_necator_H16_iJN1463.xml
   ```

2. **代替方法: ブラウザでダウンロード**
   - URL: http://bigg.ucsd.edu/models/iJN1463
   - "Download" ボタンをクリック
   - SBML形式を選択

3. **検証**
   ```bash
   python -c "import cobra; model = cobra.io.read_sbml_model('models/sbml/Cupriavidus_necator_H16_iJN1463.xml'); print(f'Genes: {len(model.genes)}, Reactions: {len(model.reactions)}')"
   ```

---

## 優先度3: Pseudomonas putida KT2440（代替PHA蓄積菌）

### 収集方法

1. **BiGG Modelsから直接ダウンロード**
   ```bash
   cd models/sbml
   wget http://bigg.ucsd.edu/static/models/iJN1462.xml -O Pseudomonas_putida_KT2440_iJN1462.xml
   ```

2. **検証**
   ```bash
   python -c "import cobra; model = cobra.io.read_sbml_model('models/sbml/Pseudomonas_putida_KT2440_iJN1462.xml'); print(f'Genes: {len(model.genes)}, Reactions: {len(model.reactions)}')"
   ```

---

## 優先度4: Bacillus subtilis 168（代替安定化菌）

### 収集方法

1. **BiGG Modelsから直接ダウンロード**
   ```bash
   cd models/sbml
   wget http://bigg.ucsd.edu/static/models/iYO844.xml -O Bacillus_subtilis_168_iYO844.xml
   ```

2. **検証**
   ```bash
   python -c "import cobra; model = cobra.io.read_sbml_model('models/sbml/Bacillus_subtilis_168_iYO844.xml'); print(f'Genes: {len(model.genes)}, Reactions: {len(model.reactions)}')"
   ```

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

## 推奨される最終的なモデルセット

理想的には、以下の6種のモデルを収集することを推奨します：

### LCP分解菌
1. ✅ Sphingobium japonicum (iJN1463) - 既存
2. 🔴 **Gordonia polyisoprenivorans (iGP1495)** - 手動収集推奨

### PHA蓄積菌
3. ⚠️ Aeromonas hydrophila (iML1515) - 実際はE. coli、名前変更推奨
4. 🔴 **Cupriavidus necator H16 (iJN1463)** - 手動収集推奨
5. 🟡 Pseudomonas putida KT2440 (iJN1462) - 手動収集可能

### 安定化菌
6. ✅ Lactobacillus plantarum (iNF517) - 既存
7. 🟡 Bacillus subtilis 168 (iYO844) - 手動収集可能

これにより、各役割について複数の選択肢を持つことができ、dFBAシミュレーションの柔軟性が向上します。

---

**作成日**: 2026年4月24日  
**更新日**: 2026年4月24日
