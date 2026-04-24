# 天然ゴム分解微生物コンソーシアム dFBA-RL プロジェクト

## 概要
天然ゴム（poly(cis-1,4-isoprene)）を分解する3種の微生物コンソーシアムを構築し、動的フラックスバランス解析（dFBA）と強化学習（RL）を統合した制御システムを実装します。

## プロジェクト構成

### フェーズ1: GEM準備（Gemini担当）
- `gemini_task_runner.py`: Gemini APIに自動的にタスクを送信
- 出力: `gemini_outputs/` ディレクトリに保存

### フェーズ2: 数理モデル設計（完了）
- MDPとしてのRL環境定義
- State/Action/Reward関数の設計

### フェーズ3: 実装（Aider担当）
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

### 3. Geminiタスクの実行
```bash
# デフォルトプロンプトで実行
python gemini_task_runner.py

# カスタムプロンプトで実行
python gemini_task_runner.py --prompt-file custom_prompt.txt --task-name custom_task
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
