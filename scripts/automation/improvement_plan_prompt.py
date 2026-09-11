import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
"""
Gemini Improvement Plan Generator
現在のSBMLファイル取得の問題を分析し、改善計画を立案
"""

IMPROVEMENT_PLAN_PROMPT = """# Task: SBML Finder パイプラインの改善計画立案

## 現状の問題分析

### 実行結果サマリー
- **成功**: 0/3 種
- **Nocardia farcinica**: ダウンロード成功、評価失敗（503エラー）
- **Burkholderia thailandensis**: ダウンロード失敗（404エラー、代替URLも全て失敗）
- **Shewanella oneidensis**: ダウンロード成功、評価失敗（XMLパースエラー）

### 具体的な問題点

1. **URL検証の不正確性**
   - Geminiが提案したURLの多くが実際には無効
   - BiGG ModelsのURL形式が古い（`bigg.ucsd.edu` vs 実際のドメイン）
   - URL検証機能が形式的で、実際のHTTPリクエストを行っていない

2. **ダウンロード失敗への対応不足**
   - Burkholderia thailandensis: 全ての代替URLが404/500エラー
   - 代替URL提案が推測ベースで、実在性が確認されていない

3. **ファイル形式の問題**
   - Shewanella oneidensis: ダウンロードしたファイルがXMLではない可能性
   - Content-Typeの検証が不足

4. **503エラーへの対応**
   - Nocardia farcinicaの評価時に503エラー
   - リトライは実装されているが、評価タスクが大きすぎる可能性

5. **モデル選定の問題**
   - 選定された3種のうち、実際にダウンロード可能なのは1種のみ
   - URL検証が不十分なまま最終選定に進んでいる

## 改善計画の要件

以下の全ての要件を満たす改善計画を提案してください：

### 必須要件
1. **実URLの事前検証**
   - Geminiに頼らず、Pythonで実際にHTTP HEADリクエストを送信
   - Content-Type、ファイルサイズを確認
   - 200 OKを返すURLのみを有効と判定

2. **既知の信頼できるソースの優先**
   - BiGG Models、ModelSEED、VMHなど主要データベースの最新URL形式を事前定義
   - データベースごとのURL構築ルールを実装

3. **段階的フォールバック戦略**
   - 第1優先: BiGG Models（最も信頼性が高い）
   - 第2優先: VMH/AGORA
   - 第3優先: ModelSEED
   - 第4優先: 文献の補足資料（最も不確実）

4. **ダウンロード後の検証**
   - XMLパース可能性の確認
   - SBMLスキーマの基本的な検証
   - ファイルサイズの妥当性チェック（最低1KB以上）

5. **評価タスクの軽量化**
   - 大きなSBMLファイルは先頭1000行のみをGeminiに送信
   - 基本統計はPython（COBRApy）で直接計算

6. **リトライ戦略の改善**
   - 503エラー時の待機時間を延長（最大16秒）
   - 評価失敗時は簡易評価（COBRApyのみ）にフォールバック

### 出力形式

以下のJSON形式で改善計画を提出してください：

```json
{
  "plan_version": "1.0",
  "improvements": [
    {
      "issue_id": "URL_VERIFICATION",
      "problem": "URL検証が形式的で実際のHTTPリクエストを行っていない",
      "solution": "requests.head()で実際にURLを検証する関数を追加",
      "implementation_steps": [
        "verify_url_with_http()関数を実装",
        "Content-Type、Status Code、ファイルサイズを確認",
        "タイムアウトを5秒に設定"
      ],
      "expected_impact": "無効なURLを事前に除外し、ダウンロード成功率を向上",
      "priority": "高"
    }
  ],
  "database_url_templates": {
    "BiGG_Models": {
      "base_url": "http://bigg.ucsd.edu/static/models/{model_id}.xml",
      "alternative_formats": [".xml.gz"],
      "verification_method": "HTTP HEAD request"
    }
  },
  "fallback_strategy": {
    "step1": "BiGG Modelsから検索",
    "step2": "失敗時はVMH/AGORAを試行",
    "step3": "失敗時はModelSEEDを試行",
    "step4": "全て失敗時は次候補の微生物種へ"
  },
  "code_changes": [
    {
      "file": "gemini_sbml_finder.py",
      "function": "_verify_url_with_http",
      "change_type": "新規追加",
      "description": "実際のHTTPリクエストでURLを検証"
    }
  ],
  "success_criteria": {
    "minimum_downloads": 3,
    "minimum_valid_xml": 3,
    "minimum_evaluation_success": 2
  },
  "risk_assessment": {
    "high_risk": [],
    "medium_risk": [],
    "low_risk": []
  },
  "estimated_improvement": {
    "download_success_rate": "現在33% → 目標100%",
    "evaluation_success_rate": "現在0% → 目標66%以上"
  }
}
```

## 重要な制約

1. **既存コードの大幅な書き換えは避ける**
   - 既存の関数構造を維持
   - 新しい検証レイヤーを追加する形で実装

2. **Gemini APIの呼び出し回数を削減**
   - URL検証はPythonで実装
   - 評価のみGeminiを使用

3. **実装可能性を重視**
   - 複雑すぎる提案は避ける
   - 段階的に実装できる計画

4. **テスト可能性**
   - 各改善が独立してテスト可能
   - ロールバックが容易

## 評価基準

提出された改善計画は以下の基準で評価されます：

1. **具体性**: 実装手順が明確か（30点）
2. **実現可能性**: 現実的に実装可能か（25点）
3. **効果**: 問題解決への寄与度（25点）
4. **リスク管理**: リスクが適切に評価されているか（10点）
5. **完全性**: 全ての問題に対処しているか（10点）

**合格基準**: 80点以上

不合格の場合は、フィードバックを受けて再提出してください。

詳細で実装可能な改善計画を提案してください。
"""
