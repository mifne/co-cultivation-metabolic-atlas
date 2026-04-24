"""
Gemini API Task Runner for GEM Preparation
Automatically sends tasks to Gemini API and saves responses.
"""

import os
import json
import time
from pathlib import Path
from google import genai


class GeminiTaskRunner:
    """Gemini APIを使用してGEM準備タスクを実行するクラス"""
    
    def _select_best_model(self) -> str:
        """
        利用可能なモデルをAPIから取得し、コストパフォーマンスが最も高いものを選択
        
        Returns:
            選択されたモデル名
        """
        try:
            print("🔍 利用可能なモデルを検索中...")
            
            # 利用可能なモデルのリストを取得
            models = self.client.models.list()
            
            # generateContentをサポートするモデルのみをフィルタ
            available_models = []
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    available_models.append({
                        'name': model.name,
                        'display_name': model.display_name,
                        'description': getattr(model, 'description', '')
                    })
            
            if not available_models:
                raise ValueError("generateContentをサポートするモデルが見つかりません")
            
            # 優先順位: Flash系 > Pro系（コストパフォーマンス重視）
            # 最新バージョンを優先
            priority_keywords = [
                'flash-2.0',
                'flash-1.5', 
                'flash',
                'pro-2.0',
                'pro-1.5',
                'pro'
            ]
            
            selected_model = None
            for keyword in priority_keywords:
                for model in available_models:
                    model_name_lower = model['name'].lower()
                    if keyword in model_name_lower:
                        selected_model = model['name']
                        print(f"✅ 選択されたモデル: {model['display_name']} ({model['name']})")
                        return selected_model
            
            # フォールバック: 最初に見つかったモデルを使用
            selected_model = available_models[0]['name']
            print(f"⚠️  デフォルトモデルを使用: {available_models[0]['display_name']} ({selected_model})")
            return selected_model
            
        except Exception as e:
            print(f"⚠️  モデル選択エラー: {str(e)}")
            print("📌 フォールバック: gemini-pro を使用")
            return 'gemini-pro'
    
    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=self.api_key)
        
        # 利用可能なモデルを取得して最適なものを選択
        self.model_name = self._select_best_model()
        
        # 出力ディレクトリの作成
        self.output_dir = Path('gemini_outputs')
        self.output_dir.mkdir(exist_ok=True)
    
    def load_prompt(self, prompt_file: str = None) -> str:
        """
        プロンプトファイルを読み込むか、デフォルトのプロンプトを返す
        
        Args:
            prompt_file: プロンプトファイルのパス（オプション）
        
        Returns:
            プロンプト文字列
        """
        if prompt_file and Path(prompt_file).exists():
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        
        # デフォルトのプロンプト（前述のGemini指示書）
        return """# Task: Genome-Scale Metabolic Model (GEM) Preparation for Natural Rubber-Degrading Consortium

## Objective
Prepare and analyze SBML-format genome-scale metabolic models (GEMs) for a three-species actinobacterial consortium capable of degrading natural rubber (poly(cis-1,4-isoprene)).

## Target Organisms
1. **Gordonia polyisoprenivorans** - Primary rubber degrader with Lcp enzyme
2. **Nocardia sp.** (e.g., N. nova or N. farcinica) - Lcp homolog, oxidative degradation
3. **Rhodococcus pyridinivorans** - High degradation capacity, cross-feeding partner

## Tasks

### Task 1: GEM Acquisition and Analysis
Search and retrieve SBML models from:
- BiGG Models database (http://bigg.ucsd.edu/)
- AGORA database
- ModelSEED
- Published literature (PubMed, supplementary materials)

For each organism:
- Download or reconstruct the SBML model
- Verify model quality (mass/charge balance, growth simulation)
- Document model source, version, and statistics (genes, reactions, metabolites)

### Task 2: Auxotrophy Design
For each species, identify gene knockout targets to create **distinct amino acid auxotrophies**:

Requirements:
- Each species should require a DIFFERENT essential amino acid
- Suggested targets:
  - Species 1: Arginine auxotrophy (knockout argininosuccinate synthase, argH)
  - Species 2: Tryptophan auxotrophy (knockout anthranilate synthase, trpE)
  - Species 3: Leucine auxotrophy (knockout 3-isopropylmalate dehydratase, leuC)

For each knockout:
- Identify the gene ID in the SBML model
- Verify essentiality via FBA simulation
- Confirm that external supplementation restores growth

### Task 3: Cross-Feeding Network Extraction
Analyze metabolic interactions related to natural rubber degradation:

1. **Identify rubber degradation pathways**:
   - Lcp enzyme reactions (if present in model, otherwise annotate manually)
   - Isoprene/isoprenoid degradation pathways
   - Beta-oxidation of degradation products

2. **Extract intermediate metabolites**:
   - Isoprenoid aldehydes (e.g., 3-methylbut-3-enal)
   - Isoprenoid ketones
   - Acetyl-CoA, propionyl-CoA
   - Other shared carbon sources

3. **Map exchange reactions**:
   - Which species can produce each intermediate?
   - Which species can consume each intermediate?
   - Create a metabolite exchange matrix

### Task 4: Model Modification for Consortium Simulation
Prepare modified SBML files:
- Add exchange reactions for cross-fed metabolites
- Implement gene knockouts for auxotrophy
- Add rubber monomer uptake reactions (if not present)
- Ensure all models use consistent metabolite IDs for shared compounds

## Deliverables
Please provide:
1. Three SBML files (one per species) with auxotrophy modifications
2. A JSON file documenting:
   - Gene knockout targets and their effects
   - Cross-feeding metabolite list with exchange reaction IDs
   - Model statistics and validation results
3. A summary report (Markdown format) explaining:
   - Rationale for auxotrophy choices
   - Predicted cross-feeding interactions
   - Any limitations or assumptions

## Output Format
```json
{
  "models": [
    {
      "species": "Gordonia polyisoprenivorans",
      "sbml_file": "gordonia_auxotroph.xml",
      "auxotrophy": "arginine",
      "knockout_gene": "argH",
      "validation": {"growth_rate_wt": 0.5, "growth_rate_ko": 0.0, "growth_rate_supplemented": 0.48}
    }
  ],
  "cross_feeding": [
    {"metabolite": "3-methylbut-3-enal", "producers": ["Gordonia"], "consumers": ["Nocardia", "Rhodococcus"]}
  ]
}
```

Please process large SBML files efficiently and provide detailed analysis."""
    
    def send_task(self, prompt: str, task_name: str = "gem_preparation") -> dict:
        """
        Gemini APIにタスクを送信し、レスポンスを取得
        
        Args:
            prompt: 送信するプロンプト
            task_name: タスク名（ファイル保存用）
        
        Returns:
            レスポンスデータを含む辞書
        """
        print(f"🚀 Geminiにタスクを送信中: {task_name}")
        print(f"📝 プロンプト長: {len(prompt)} 文字")
        
        try:
            # Gemini APIにリクエスト送信
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            # レスポンスの処理
            result = {
                'task_name': task_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'prompt': prompt,
                'response_text': response.text,
                'status': 'success'
            }
            
            print(f"✅ レスポンス受信完了")
            print(f"📊 レスポンス長: {len(response.text)} 文字")
            
            return result
            
        except Exception as e:
            print(f"❌ エラー発生: {str(e)}")
            return {
                'task_name': task_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'prompt': prompt,
                'error': str(e),
                'status': 'failed'
            }
    
    def save_response(self, result: dict):
        """
        レスポンスをファイルに保存
        
        Args:
            result: send_taskの戻り値
        """
        task_name = result['task_name']
        timestamp = result['timestamp'].replace(':', '-').replace(' ', '_')
        
        # JSONファイルとして保存
        json_file = self.output_dir / f"{task_name}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"💾 JSONファイル保存: {json_file}")
        
        # レスポンステキストをMarkdownとして保存
        if result['status'] == 'success':
            md_file = self.output_dir / f"{task_name}_{timestamp}_response.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(f"# Gemini Response: {task_name}\n\n")
                f.write(f"**Timestamp:** {result['timestamp']}\n\n")
                f.write("---\n\n")
                f.write(result['response_text'])
            print(f"📄 Markdownファイル保存: {md_file}")
    
    def run_full_pipeline(self, prompt_file: str = None):
        """
        完全なパイプラインを実行
        
        Args:
            prompt_file: カスタムプロンプトファイル（オプション）
        """
        print("=" * 60)
        print("🧬 Gemini GEM Preparation Task Runner")
        print("=" * 60)
        
        # プロンプトの読み込み
        prompt = self.load_prompt(prompt_file)
        
        # タスクの送信
        result = self.send_task(prompt, task_name="gem_preparation")
        
        # レスポンスの保存
        self.save_response(result)
        
        print("=" * 60)
        print("✨ タスク完了")
        print(f"📁 出力ディレクトリ: {self.output_dir.absolute()}")
        print("=" * 60)
        
        return result


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Gemini APIを使用してGEM準備タスクを実行'
    )
    parser.add_argument(
        '--prompt-file',
        type=str,
        help='カスタムプロンプトファイルのパス',
        default=None
    )
    parser.add_argument(
        '--task-name',
        type=str,
        help='タスク名（デフォルト: gem_preparation）',
        default='gem_preparation'
    )
    
    args = parser.parse_args()
    
    # タスクランナーの初期化と実行
    runner = GeminiTaskRunner()
    
    if args.prompt_file:
        prompt = runner.load_prompt(args.prompt_file)
        result = runner.send_task(prompt, task_name=args.task_name)
        runner.save_response(result)
    else:
        runner.run_full_pipeline()


if __name__ == '__main__':
    main()
