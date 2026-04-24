"""
Gemini SBML Finder - 天然ゴム分解微生物のSBMLモデル検索・取得
サブエージェントによる思考→検索→ダウンロード→評価の完全自動化
"""

import os
import json
import time
from pathlib import Path
from google import genai
import requests
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET


class GeminiSBMLFinder:
    """Gemini APIを使用してSBMLモデルを検索・取得・評価するクラス"""
    
    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = 'gemini-2.5-flash'
        
        # 出力ディレクトリの作成
        self.output_dir = Path('models/sbml')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path('gemini_outputs')
        self.log_dir.mkdir(exist_ok=True)
    
    def _call_gemini(self, prompt: str, task_name: str = "sbml_search") -> str:
        """
        Gemini APIを呼び出す
        
        Args:
            prompt: プロンプト
            task_name: タスク名
        
        Returns:
            レスポンステキスト
        """
        print(f"🤖 Gemini呼び出し中: {task_name}")
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            # ログ保存
            timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
            log_file = self.log_dir / f"{task_name}_{timestamp}.json"
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'task': task_name,
                    'timestamp': timestamp,
                    'prompt': prompt,
                    'response': response.text
                }, f, ensure_ascii=False, indent=2)
            
            return response.text
            
        except Exception as e:
            print(f"❌ Geminiエラー: {e}")
            return ""
    
    def step1_select_organisms(self) -> Dict[str, str]:
        """
        ステップ1: サブエージェントによる微生物選定
        
        Returns:
            {species_name: rationale}
        """
        print("\n" + "="*60)
        print("📋 ステップ1: 微生物種の選定")
        print("="*60)
        
        prompt = """# Task: 天然ゴム分解コンソーシアムの微生物種選定

## 目的
天然ゴム（poly(cis-1,4-isoprene)）を効率的に分解し、付加価値物質（PHA）を生産する3種の微生物コンソーシアムを構築するため、最適な微生物種を選定してください。

## 必須の3つの役割
1. **LCP分解菌**: Lcp酵素を保有し、天然ゴムポリマーを低分子イソプレノイドに分解
2. **PHA蓄積菌**: イソプレノイド分解産物を取り込み、ポリヒドロキシアルカノエート（PHA）に変換・蓄積
3. **安定化菌**: コンソーシアム全体の代謝バランスを維持し、pH調整や副産物処理を担当

## 要件
1. **各種が明確な役割を持つこと**（上記3役割を分担）
2. **放線菌に限定しない**（グラム陰性菌、他の門も可）
3. **代謝的相補性があること**（異なる栄養要求性、クロスフィーディング）
4. **ゲノム規模代謝モデル（GEM）が公開されていること**
   - BiGG Models (http://bigg.ucsd.edu/)
   - AGORA database
   - ModelSEED
   - 文献の補足資料（SBML形式で入手可能）

## 候補微生物（参考）
### LCP分解菌候補
- Gordonia polyisoprenivorans (放線菌)
- Nocardia sp. (放線菌)
- Rhodococcus sp. (放線菌)

### PHA蓄積菌候補
- Cupriavidus necator (β-プロテオバクテリア、PHA生産のモデル株)
- Pseudomonas putida (γ-プロテオバクテリア、PHA生産能)
- Burkholderia sp. (β-プロテオバクテリア)

### 安定化菌候補
- Escherichia coli (γ-プロテオバクテリア、代謝工学のモデル株)
- Bacillus subtilis (ファーミキューテス、pH調整、バイオフィルム形成)
- Pseudomonas fluorescens (γ-プロテオバクテリア、多様な代謝能)

## 出力形式
以下のJSON形式で3種を選定し、各種について詳細な根拠を示してください：

```json
{
  "selected_organisms": [
    {
      "species_name": "Gordonia polyisoprenivorans",
      "strain": "VH2",
      "phylum": "Actinobacteria",
      "consortium_role": "LCP分解菌",
      "rationale": "Lcp1/Lcp2酵素を保有し、天然ゴムを効率的に低分子イソプレノイドに分解。",
      "key_capabilities": ["天然ゴム分解", "イソプレノイド生成"],
      "expected_auxotrophy": "アルギニン",
      "gem_availability": "文献 (PMID: 35149306) にSBMLモデルあり",
      "gem_quality_estimate": "高"
    },
    {
      "species_name": "Cupriavidus necator",
      "strain": "H16",
      "phylum": "Proteobacteria",
      "consortium_role": "PHA蓄積菌",
      "rationale": "イソプレノイド分解産物（アセチルCoA等）からPHAを高効率で合成・蓄積。",
      "key_capabilities": ["PHA合成", "炭素源多様性"],
      "expected_auxotrophy": "トリプトファン",
      "gem_availability": "BiGG Models (iJN1463) またはModelSEED",
      "gem_quality_estimate": "非常に高"
    },
    {
      "species_name": "Pseudomonas putida",
      "strain": "KT2440",
      "phylum": "Proteobacteria",
      "consortium_role": "安定化菌",
      "rationale": "多様な有機酸・芳香族化合物を代謝し、副産物を処理。pH安定化に寄与。",
      "key_capabilities": ["副産物代謝", "pH調整", "バイオフィルム形成"],
      "expected_auxotrophy": "ロイシン",
      "gem_availability": "BiGG Models (iJN746) またはModelSEED",
      "gem_quality_estimate": "非常に高"
    }
  ],
  "consortium_design_rationale": "LCP分解菌がゴムを分解→PHA蓄積菌が中間体を資源化→安定化菌が系全体を維持、という明確な役割分担により、効率的かつ安定したコンソーシアムを構築。",
  "expected_cross_feeding": [
    "イソプレノイド中間体 (Gordonia → Cupriavidus)",
    "有機酸 (Cupriavidus → Pseudomonas)",
    "アミノ酸 (相互補完)"
  ],
  "metabolic_complementarity": "各種が異なるアミノ酸栄養要求性を持ち、相互依存関係を構築。"
}
```

## 重要な判断基準
1. **役割の明確性**: 3つの役割（LCP分解、PHA蓄積、安定化）が明確に分担されていること
2. **GEM入手可能性**: 必ず公開データベースまたは文献から**SBML形式で**入手可能であること
3. **代謝的相補性**: 異なる栄養要求性を持ち、クロスフィーディングが成立すること
4. **実験的実績**: 文献で報告されている実績のある種を優先
5. **PHA生産能**: PHA蓄積菌は高いPHA生産能を持つこと

思考プロセスを明確に示し、最終的な選定理由を詳しく説明してください。"""

        response = self._call_gemini(prompt, "organism_selection")
        
        # JSONを抽出
        try:
            # ```json ... ``` ブロックを探す
            json_start = response.find('```json')
            if json_start != -1:
                json_start = response.find('\n', json_start) + 1
                json_end = response.find('```', json_start)
                json_str = response[json_start:json_end].strip()
                selection_data = json.loads(json_str)
                
                print("\n✅ 選定された微生物:")
                organisms = {}
                for org in selection_data.get('selected_organisms', []):
                    species = org['species_name']
                    print(f"  - {species}")
                    print(f"    根拠: {org['rationale']}")
                    organisms[species] = org
                
                # 選定結果を保存
                selection_file = self.log_dir / 'organism_selection.json'
                with open(selection_file, 'w', encoding='utf-8') as f:
                    json.dump(selection_data, f, ensure_ascii=False, indent=2)
                
                return organisms
            else:
                print("⚠️  JSON形式のレスポンスが見つかりませんでした")
                return {}
                
        except Exception as e:
            print(f"❌ JSON解析エラー: {e}")
            return {}
    
    def step2_search_sbml_models(self, organisms: Dict[str, str]) -> Dict[str, List[Dict]]:
        """
        ステップ2: SBMLモデルの検索
        
        Args:
            organisms: 選定された微生物の辞書
        
        Returns:
            {species_name: [model_info]}
        """
        print("\n" + "="*60)
        print("🔍 ステップ2: SBMLモデルの検索")
        print("="*60)
        
        all_models = {}
        
        for species_name, org_info in organisms.items():
            print(f"\n📚 {species_name} のモデルを検索中...")
            
            prompt = f"""# Task: {species_name} のゲノム規模代謝モデル（GEM）検索

## 対象微生物
- 種名: {species_name}
- 株: {org_info.get('strain', '不明')}

## 検索先データベース
1. **BiGG Models** (http://bigg.ucsd.edu/)
   - URLパターン: http://bigg.ucsd.edu/models/[model_id]
   - ダウンロードURL: http://bigg.ucsd.edu/static/models/[model_id].xml

2. **AGORA** (https://www.vmh.life/)
   - 腸内細菌中心だが一部放線菌も含む

3. **ModelSEED** (https://modelseed.org/)
   - KBase統合

4. **文献の補足資料**
   - PubMed検索: "{species_name} genome-scale metabolic model"
   - 補足資料のSBMLファイル

## 出力形式
以下のJSON形式で、見つかったすべてのモデル情報を返してください：

```json
{{
  "models_found": [
    {{
      "model_id": "iGP123",
      "source": "BiGG Models",
      "download_url": "http://bigg.ucsd.edu/static/models/iGP123.xml",
      "direct_download": true,
      "quality_score": 9,
      "notes": "高品質、キュレーション済み"
    }},
    {{
      "model_id": "draft_model",
      "source": "文献 (PMID: 12345678)",
      "download_url": "https://doi.org/10.xxxx/supplementary_file.xml",
      "direct_download": false,
      "quality_score": 6,
      "notes": "ドラフトモデル、要検証"
    }}
  ],
  "search_summary": "BiGG Modelsで1件、文献で1件見つかりました。",
  "recommendation": "BiGG ModelsのiGP123を推奨（高品質）"
}}
```

## 重要
- **実在するURLのみ**を記載してください
- ダウンロード可能性を明記してください
- モデルの品質を1-10で評価してください
- 見つからない場合は、代替案（近縁種のモデル）を提案してください

実際にデータベースを検索し、具体的なURLを提供してください。"""

            response = self._call_gemini(prompt, f"sbml_search_{species_name.replace(' ', '_')}")
            
            # JSONを抽出
            try:
                json_start = response.find('```json')
                if json_start != -1:
                    json_start = response.find('\n', json_start) + 1
                    json_end = response.find('```', json_start)
                    json_str = response[json_start:json_end].strip()
                    search_data = json.loads(json_str)
                    
                    models = search_data.get('models_found', [])
                    all_models[species_name] = models
                    
                    print(f"  ✅ {len(models)}件のモデルが見つかりました")
                    for model in models:
                        print(f"    - {model['model_id']} ({model['source']})")
                        print(f"      品質: {model['quality_score']}/10")
                
            except Exception as e:
                print(f"  ❌ JSON解析エラー: {e}")
                all_models[species_name] = []
        
        # 検索結果を保存
        search_file = self.log_dir / 'sbml_search_results.json'
        with open(search_file, 'w', encoding='utf-8') as f:
            json.dump(all_models, f, ensure_ascii=False, indent=2)
        
        return all_models
    
    def step2_5_filter_organisms(self, organisms: Dict[str, str], models_info: Dict[str, List[Dict]]) -> Dict[str, str]:
        """
        ステップ2.5: 選定された微生物とモデル情報をフィルタリング
        
        Args:
            organisms: 選定された微生物の辞書
            models_info: モデル情報の辞書
        
        Returns:
            フィルタリング後の微生物辞書
        """
        print("\n" + "="*60)
        print("🔍 ステップ2.5: 微生物とモデルのフィルタリング")
        print("="*60)
        
        # 各微生物のモデル入手可能性をチェック
        filtered = {}
        
        for species_name, org_info in organisms.items():
            models = models_info.get(species_name, [])
            
            if not models:
                print(f"\n⚠️  {species_name}: モデルが見つかりませんでした")
                print(f"  役割: {org_info.get('consortium_role', '不明')}")
                print(f"  → 除外候補")
                continue
            
            # 最高品質のモデルを確認
            best_model = max(models, key=lambda m: m.get('quality_score', 0))
            quality = best_model.get('quality_score', 0)
            
            print(f"\n✅ {species_name}")
            print(f"  役割: {org_info.get('consortium_role', '不明')}")
            print(f"  最良モデル: {best_model['model_id']}")
            print(f"  品質スコア: {quality}/10")
            print(f"  ダウンロード可能: {best_model.get('direct_download', False)}")
            
            if quality >= 6:
                filtered[species_name] = org_info
                print(f"  → 採用")
            else:
                print(f"  → 品質不足により除外")
        
        # 役割の重複チェック
        roles = [org.get('consortium_role', '') for org in filtered.values()]
        role_counts = {role: roles.count(role) for role in set(roles)}
        
        print(f"\n📊 役割分担:")
        for role, count in role_counts.items():
            print(f"  {role}: {count}種")
        
        # 3つの役割が揃っているか確認
        required_roles = ['LCP分解菌', 'PHA蓄積菌', '安定化菌']
        missing_roles = [r for r in required_roles if r not in roles]
        
        if missing_roles:
            print(f"\n⚠️  不足している役割: {', '.join(missing_roles)}")
            print(f"  → Geminiに代替微生物の提案を依頼します")
            
            # Geminiに代替案を求める
            alt_prompt = f"""# Task: コンソーシアム微生物の代替案提案

以下の役割が不足しています：
{', '.join(missing_roles)}

現在選定されている微生物：
{json.dumps([{'species': k, 'role': v.get('consortium_role')} for k, v in filtered.items()], ensure_ascii=False, indent=2)}

## 要求
不足している役割を担う代替微生物を提案してください。
- **GEMが確実に入手可能**であること（BiGG Models、ModelSEED、文献）
- 既存の微生物と代謝的相補性があること

JSON形式で回答：
```json
{{
  "alternative_organisms": [
    {{
      "species_name": "種名",
      "strain": "株名",
      "consortium_role": "役割",
      "rationale": "理由",
      "gem_source": "BiGG Models / ModelSEED / 文献"
    }}
  ]
}}
```"""
            
            alt_response = self._call_gemini(alt_prompt, "alternative_organisms")
            print(f"\n📝 代替案レスポンス受信")
            
            # 代替案のJSONを抽出（簡略化のため、ここでは警告のみ）
            try:
                json_start = alt_response.find('```json')
                if json_start != -1:
                    json_start = alt_response.find('\n', json_start) + 1
                    json_end = alt_response.find('```', json_start)
                    json_str = alt_response[json_start:json_end].strip()
                    alt_data = json.loads(json_str)
                    print(f"  提案された代替微生物: {len(alt_data.get('alternative_organisms', []))}種")
            except Exception as e:
                print(f"  ⚠️  代替案の解析に失敗: {e}")
        else:
            print(f"\n✅ 3つの役割が揃っています")
        
        return filtered
    
    def step3_download_sbml_files(self, models_info: Dict[str, List[Dict]]) -> Dict[str, Optional[Path]]:
        """
        ステップ3: SBMLファイルのダウンロード
        
        Args:
            models_info: モデル情報の辞書
        
        Returns:
            {species_name: downloaded_file_path}
        """
        print("\n" + "="*60)
        print("⬇️  ステップ3: SBMLファイルのダウンロード")
        print("="*60)
        
        downloaded_files = {}
        
        for species_name, models in models_info.items():
            if not models:
                print(f"\n⚠️  {species_name}: モデルが見つかりませんでした")
                downloaded_files[species_name] = None
                continue
            
            # 最高品質のモデルを選択
            best_model = max(models, key=lambda m: m.get('quality_score', 0))
            
            # safe_nameを先に定義
            safe_name = species_name.replace(' ', '_').replace('.', '')
            
            print(f"\n📥 {species_name}: {best_model['model_id']} をダウンロード中...")
            print(f"  ソース: {best_model['source']}")
            print(f"  URL: {best_model['download_url']}")
            
            try:
                # ダウンロード試行
                response = requests.get(best_model['download_url'], timeout=30)
                response.raise_for_status()
                
                # ファイル保存
                file_path = self.output_dir / f"{safe_name}_{best_model['model_id']}.xml"
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"  ✅ ダウンロード成功: {file_path}")
                downloaded_files[species_name] = file_path
                
            except Exception as e:
                print(f"  ❌ ダウンロード失敗: {e}")
                
                # Geminiに代替案を求める
                print(f"  🤖 Geminiに代替ダウンロード方法を問い合わせ中...")
                
                alt_prompt = f"""# Task: {species_name} のSBMLモデル代替取得方法

以下のURLからのダウンロードに失敗しました：
- URL: {best_model['download_url']}
- エラー: {str(e)}

## 代替案を提案してください
1. 別のダウンロードURL（ミラーサイト、GitHub等）
2. 手動ダウンロード手順
3. 近縁種のモデルで代用可能な場合はその情報

JSON形式で回答してください：
```json
{{
  "alternative_urls": ["url1", "url2"],
  "manual_instructions": "手順...",
  "substitute_model": {{
    "species": "近縁種名",
    "url": "URL"
  }}
}}
```"""
                
                alt_response = self._call_gemini(alt_prompt, f"alternative_download_{safe_name}")
                print(f"  📝 代替案: {alt_response[:200]}...")
                
                downloaded_files[species_name] = None
        
        return downloaded_files
    
    def step4_evaluate_sbml_files(self, downloaded_files: Dict[str, Optional[Path]]) -> Dict[str, Dict]:
        """
        ステップ4: SBMLファイルの評価
        
        Args:
            downloaded_files: ダウンロードされたファイルのパス
        
        Returns:
            {species_name: evaluation_result}
        """
        print("\n" + "="*60)
        print("📊 ステップ4: SBMLファイルの評価")
        print("="*60)
        
        evaluations = {}
        
        for species_name, file_path in downloaded_files.items():
            if file_path is None or not file_path.exists():
                print(f"\n⚠️  {species_name}: ファイルが存在しません")
                evaluations[species_name] = {'status': 'missing'}
                continue
            
            print(f"\n🔬 {species_name} を評価中...")
            
            try:
                # 基本的なXML検証
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # 統計情報を収集
                stats = {
                    'file_size': file_path.stat().st_size,
                    'xml_valid': True,
                    'root_tag': root.tag
                }
                
                print(f"  ✅ XML検証: OK")
                print(f"  📏 ファイルサイズ: {stats['file_size']:,} bytes")
                
                # Geminiに詳細評価を依頼
                with open(file_path, 'r', encoding='utf-8') as f:
                    sbml_content = f.read()
                
                # 大きすぎる場合は先頭部分のみ
                if len(sbml_content) > 50000:
                    sbml_sample = sbml_content[:50000] + "\n... (truncated)"
                else:
                    sbml_sample = sbml_content
                
                eval_prompt = f"""# Task: SBMLモデルの詳細評価

## 対象
- 種名: {species_name}
- ファイル: {file_path.name}
- サイズ: {stats['file_size']:,} bytes

## SBMLファイル（サンプル）
```xml
{sbml_sample}
```

## 評価項目
1. **モデル統計**
   - 遺伝子数
   - 反応数
   - 代謝物数
   - コンパートメント数

2. **品質チェック**
   - バイオマス反応の有無
   - 交換反応の数
   - 質量・電荷バランス（推定）

3. **天然ゴム分解関連**
   - Lcp酵素関連反応の有無
   - イソプレノイド代謝経路
   - ベータ酸化経路

4. **栄養要求性候補**
   - アミノ酸合成経路の完全性
   - 推奨される栄養要求性遺伝子ノックアウト

## 出力形式
```json
{{
  "statistics": {{
    "genes": 1234,
    "reactions": 2345,
    "metabolites": 1567,
    "compartments": 3
  }},
  "quality": {{
    "biomass_reaction": true,
    "exchange_reactions": 123,
    "quality_score": 8
  }},
  "rubber_degradation": {{
    "lcp_enzyme": false,
    "isoprenoid_pathway": true,
    "beta_oxidation": true
  }},
  "auxotrophy_candidates": [
    {{"amino_acid": "arginine", "target_gene": "argH", "confidence": "high"}},
    {{"amino_acid": "tryptophan", "target_gene": "trpE", "confidence": "medium"}}
  ],
  "overall_assessment": "高品質なモデル。ゴム分解研究に適している。",
  "recommendations": "argH遺伝子のノックアウトでアルギニン栄養要求株を作成可能"
}}
```

詳細に評価してください。"""
                
                eval_response = self._call_gemini(eval_prompt, f"evaluation_{species_name.replace(' ', '_')}")
                
                # JSON抽出
                try:
                    json_start = eval_response.find('```json')
                    if json_start != -1:
                        json_start = eval_response.find('\n', json_start) + 1
                        json_end = eval_response.find('```', json_start)
                        json_str = eval_response[json_start:json_end].strip()
                        eval_data = json.loads(json_str)
                        
                        evaluations[species_name] = {
                            'status': 'success',
                            'file_path': str(file_path),
                            **eval_data
                        }
                        
                        print(f"  📊 統計:")
                        print(f"    遺伝子: {eval_data['statistics']['genes']}")
                        print(f"    反応: {eval_data['statistics']['reactions']}")
                        print(f"    代謝物: {eval_data['statistics']['metabolites']}")
                        print(f"  ⭐ 品質スコア: {eval_data['quality']['quality_score']}/10")
                        
                    else:
                        evaluations[species_name] = {
                            'status': 'partial',
                            'file_path': str(file_path),
                            'basic_stats': stats
                        }
                        
                except Exception as e:
                    print(f"  ⚠️  評価JSON解析エラー: {e}")
                    evaluations[species_name] = {
                        'status': 'partial',
                        'file_path': str(file_path),
                        'basic_stats': stats
                    }
                
            except Exception as e:
                print(f"  ❌ 評価エラー: {e}")
                evaluations[species_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # 評価結果を保存
        eval_file = self.log_dir / 'sbml_evaluation.json'
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(evaluations, f, ensure_ascii=False, indent=2)
        
        return evaluations
    
    def run_full_pipeline(self):
        """完全なパイプラインを実行"""
        print("\n" + "="*60)
        print("🧬 Gemini SBML Finder - 完全自動パイプライン")
        print("="*60)
        
        # ステップ1: 微生物選定
        organisms = self.step1_select_organisms()
        if not organisms:
            print("\n❌ 微生物選定に失敗しました")
            return
        
        # ステップ2: SBMLモデル検索
        models_info = self.step2_search_sbml_models(organisms)
        
        # ステップ2.5: フィルタリング
        filtered_organisms = self.step2_5_filter_organisms(organisms, models_info)
        
        if len(filtered_organisms) < 3:
            print(f"\n⚠️  フィルタリング後の微生物数が不足: {len(filtered_organisms)}/3")
            print(f"  → パイプラインを中断します")
            return
        
        # フィルタリング後のモデル情報のみを使用
        filtered_models_info = {k: v for k, v in models_info.items() if k in filtered_organisms}
        
        # ステップ3: ダウンロード
        downloaded_files = self.step3_download_sbml_files(filtered_models_info)
        
        # ステップ4: 評価
        evaluations = self.step4_evaluate_sbml_files(downloaded_files)
        
        # 最終レポート
        print("\n" + "="*60)
        print("📋 最終レポート")
        print("="*60)
        
        success_count = sum(1 for e in evaluations.values() if e.get('status') == 'success')
        print(f"\n✅ 成功: {success_count}/3 種")
        
        for species, eval_data in evaluations.items():
            status = eval_data.get('status', 'unknown')
            print(f"\n{species}:")
            print(f"  ステータス: {status}")
            if status == 'success':
                print(f"  ファイル: {eval_data['file_path']}")
                print(f"  品質: {eval_data.get('quality', {}).get('quality_score', 'N/A')}/10")
        
        print(f"\n📁 SBMLファイル保存先: {self.output_dir.absolute()}")
        print(f"📁 ログ保存先: {self.log_dir.absolute()}")
        print("\n" + "="*60)
        print("✨ パイプライン完了")
        print("="*60)


def main():
    """メイン実行関数"""
    finder = GeminiSBMLFinder()
    finder.run_full_pipeline()


if __name__ == '__main__':
    main()
