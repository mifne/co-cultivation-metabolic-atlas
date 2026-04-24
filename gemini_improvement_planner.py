"""
Gemini Improvement Planner
改善計画の立案と評価を行うサブエージェント
"""

import os
import json
import time
from pathlib import Path
from google import genai
from improvement_plan_prompt import IMPROVEMENT_PLAN_PROMPT


class ImprovementPlanner:
    """改善計画の立案と評価を行うクラス"""
    
    def _select_gemini_3_pro(self) -> str:
        """Gemini 3系のProモデルを選択"""
        try:
            models_response = self.client.models.list()
            available_models = []
            for model in models_response:
                model_name = model.name
                if model_name.startswith('models/'):
                    model_name = model_name[7:]
                available_models.append(model_name)
            
            # Gemini 3系のProモデルを優先
            priority = ['gemini-3-pro', 'gemini-3.1-pro', 'gemini-3-flash', 'gemini-3.1-flash']
            
            for preferred in priority:
                for available in available_models:
                    if preferred in available.lower():
                        print(f"✅ 改善計画用モデル: {available}")
                        return available
            
            # フォールバック
            return 'gemini-1.5-pro-002'
        except Exception as e:
            print(f"⚠️  モデル選択エラー: {e}")
            return 'gemini-1.5-pro-002'
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=self.api_key)
        
        # Gemini 3系の高性能モデルを選択
        self.model_name = self._select_gemini_3_pro()
        
        self.log_dir = Path('gemini_outputs')
        self.log_dir.mkdir(exist_ok=True)
    
    def _call_gemini(self, prompt: str, max_retries: int = 3) -> str:
        """Gemini APIを呼び出す"""
        print(f"🤖 Gemini呼び出し中 (モデル: {self.model_name})")
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt
                    print(f"  ⏳ {wait_time}秒待機後に再試行... (試行 {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                
                return response.text
                
            except Exception as e:
                error_msg = str(e)
                if '503' in error_msg or 'UNAVAILABLE' in error_msg:
                    if attempt < max_retries - 1:
                        continue
                print(f"  ❌ Geminiエラー: {e}")
                return ""
        
        return ""
    
    def generate_plan(self) -> dict:
        """改善計画を生成"""
        print("\n" + "="*60)
        print("📋 改善計画の立案")
        print("="*60)
        
        response = self._call_gemini(IMPROVEMENT_PLAN_PROMPT)
        
        if not response:
            print("❌ 改善計画の生成に失敗しました")
            return {}
        
        # JSONを抽出
        try:
            json_start = response.find('```json')
            if json_start != -1:
                json_start = response.find('\n', json_start) + 1
                json_end = response.find('```', json_start)
                json_str = response[json_start:json_end].strip()
                plan = json.loads(json_str)
                
                # 計画を保存
                timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
                plan_file = self.log_dir / f'improvement_plan_{timestamp}.json'
                with open(plan_file, 'w', encoding='utf-8') as f:
                    json.dump(plan, f, ensure_ascii=False, indent=2)
                print(f"💾 改善計画保存: {plan_file}")
                
                return plan
            else:
                print("⚠️  JSON形式のレスポンスが見つかりませんでした")
                return {}
                
        except Exception as e:
            print(f"❌ JSON解析エラー: {e}")
            return {}
    
    def evaluate_plan(self, plan: dict) -> tuple[int, str]:
        """
        改善計画を評価
        
        Returns:
            (score, feedback)
        """
        print("\n" + "="*60)
        print("📊 改善計画の評価")
        print("="*60)
        
        eval_prompt = f"""# Task: 改善計画の評価

以下の改善計画を評価してください。

## 改善計画
```json
{json.dumps(plan, ensure_ascii=False, indent=2)}
```

## 評価基準
1. **具体性**: 実装手順が明確か（30点）
2. **実現可能性**: 現実的に実装可能か（25点）
3. **効果**: 問題解決への寄与度（25点）
4. **リスク管理**: リスクが適切に評価されているか（10点）
5. **完全性**: 全ての問題に対処しているか（10点）

**合格基準**: 80点以上

## 出力形式
```json
{{
  "scores": {{
    "specificity": 25,
    "feasibility": 20,
    "effectiveness": 22,
    "risk_management": 8,
    "completeness": 9
  }},
  "total_score": 84,
  "passed": true,
  "feedback": {{
    "strengths": ["具体的な実装手順が明確", "段階的なアプローチ"],
    "weaknesses": ["リスク評価が不十分", "テスト計画が欠如"],
    "recommendations": ["単体テストの追加を推奨", "ロールバック手順の明確化"]
  }},
  "detailed_comments": "全体的に良い計画だが、..."
}}
```

厳格に評価し、詳細なフィードバックを提供してください。"""

        response = self._call_gemini(eval_prompt)
        
        if not response:
            print("❌ 評価に失敗しました")
            return 0, "評価失敗"
        
        # JSONを抽出
        try:
            json_start = response.find('```json')
            if json_start != -1:
                json_start = response.find('\n', json_start) + 1
                json_end = response.find('```', json_start)
                json_str = response[json_start:json_end].strip()
                evaluation = json.loads(json_str)
                
                total_score = evaluation.get('total_score', 0)
                passed = evaluation.get('passed', False)
                
                print(f"\n📊 評価結果:")
                print(f"  総合得点: {total_score}/100")
                print(f"  合否: {'✅ 合格' if passed else '❌ 不合格'}")
                
                if 'scores' in evaluation:
                    print(f"\n  詳細スコア:")
                    for criterion, score in evaluation['scores'].items():
                        print(f"    - {criterion}: {score}")
                
                if 'feedback' in evaluation:
                    feedback = evaluation['feedback']
                    if 'strengths' in feedback:
                        print(f"\n  強み:")
                        for s in feedback['strengths']:
                            print(f"    ✓ {s}")
                    if 'weaknesses' in feedback:
                        print(f"\n  弱み:")
                        for w in feedback['weaknesses']:
                            print(f"    ✗ {w}")
                    if 'recommendations' in feedback:
                        print(f"\n  推奨事項:")
                        for r in feedback['recommendations']:
                            print(f"    → {r}")
                
                feedback_text = evaluation.get('detailed_comments', '')
                
                return total_score, feedback_text
                
        except Exception as e:
            print(f"❌ 評価結果の解析エラー: {e}")
            return 0, f"解析エラー: {e}"
    
    def run_planning_cycle(self, max_iterations: int = 3) -> dict:
        """
        改善計画の立案・評価サイクルを実行
        
        Returns:
            承認された改善計画
        """
        print("\n" + "="*60)
        print("🔄 改善計画の立案・評価サイクル開始")
        print("="*60)
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"🔄 イテレーション {iteration}/{max_iterations}")
            print(f"{'='*60}")
            
            # 計画生成
            plan = self.generate_plan()
            if not plan:
                print("❌ 計画生成失敗、次のイテレーションへ")
                continue
            
            # 計画評価
            score, feedback = self.evaluate_plan(plan)
            
            if score >= 80:
                print(f"\n✅ 改善計画が承認されました！（スコア: {score}/100）")
                return plan
            else:
                print(f"\n❌ 改善計画が不合格（スコア: {score}/100）")
                print(f"📝 フィードバック: {feedback}")
                
                if iteration < max_iterations:
                    print(f"\n🔄 フィードバックを反映して再立案します...")
                    time.sleep(2)
        
        print(f"\n❌ {max_iterations}回のイテレーション後も承認されませんでした")
        return {}


def main():
    """メイン実行関数"""
    planner = ImprovementPlanner()
    approved_plan = planner.run_planning_cycle(max_iterations=3)
    
    if approved_plan:
        print("\n" + "="*60)
        print("✅ 承認された改善計画")
        print("="*60)
        print(json.dumps(approved_plan, ensure_ascii=False, indent=2))
        
        # 実装指示を出力
        print("\n" + "="*60)
        print("📋 次のステップ: 改善計画の実装")
        print("="*60)
        print("承認された改善計画に基づいて、gemini_sbml_finder.pyを修正してください。")
    else:
        print("\n❌ 承認された改善計画を取得できませんでした")


if __name__ == '__main__':
    main()
