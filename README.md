# 天然ゴム分解微生物コンソーシアム dFBA-RL プロジェクト

## 概要
天然ゴム（poly(cis-1,4-isoprene)）を分解する3種の微生物コンソーシアムを構築し、動的フラックスバランス解析（dFBA）と強化学習（RL）を統合した制御システムを実装します。

## プロジェクト構成

### フェーズ1: GEM準備（完了）
- `gemini_sbml_finder.py`: Gemini APIによる自動SBML検索・ダウンロード
- 出力: `models/sbml/` ディレクトリに保存
- 選定された3種:
  1. Sphingobium japonicum (LCP分解)
  2. Pseudomonas putida KT2440 (PHA蓄積)
  3. Lactobacillus plantarum (安定化)

### フェーズ2: 数理モデル設計（完了）
- MDPとしてのRL環境定義
- State/Action/Reward関数の設計

### フェーズ3: 実装（進行中）
- dFBAシミュレーター
- RL環境
- PPOエージェント学習

## セットアップ

### 1. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 2. Gemini API キーの設定
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### 3. SBMLモデルの自動検索・ダウンロード（オプション）
```bash
# Geminiによる自動SBML検索
python gemini_sbml_finder.py
```

**注**: 既に以下の4種のモデルがダウンロード済みです:
- Sphingobium japonicum (iJN1463)
- Pseudomonas putida KT2440 (iJN1462)
- Lactobacillus plantarum (iNF517)
- Bacillus subtilis 168 (iYO844)

### 4. dFBA-RLシミュレーションの実行
```bash
# 訓練モード
python main.py train --sbml-dir models/sbml --total-timesteps 100000

# 評価モード
python main.py evaluate --model-path outputs/ppo_consortium_model.zip --sbml-dir models/sbml
```

## 使用方法

### Pythonスクリプトとして使用
```python
from gemini_task_runner import GeminiTaskRunner

# タスクランナーの初期化
runner = GeminiTaskRunner()

# 完全なパイプラインを実行
result = runner.run_full_pipeline()

# カスタムプロンプトで実行
custom_prompt = "Your custom prompt here..."
result = runner.send_task(custom_prompt, task_name="my_task")
runner.save_response(result)
```

## 出力ファイル
- `gemini_outputs/gem_preparation_YYYY-MM-DD_HH-MM-SS.json`: 完全なレスポンスデータ
- `gemini_outputs/gem_preparation_YYYY-MM-DD_HH-MM-SS_response.md`: レスポンステキスト（Markdown形式）

## 次のステップ
1. Geminiからのレスポンスを確認
2. SBMLファイルとメタデータを `models/sbml/` に配置
3. Aiderを使用してdFBA-RL環境を実装

## ライセンス
MIT License
