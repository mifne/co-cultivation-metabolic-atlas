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
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


class GeminiSBMLFinder:
    """Gemini APIを使用してSBMLモデルを検索・取得・評価するクラス"""
    
    # データベースURLテンプレート
    DATABASE_URL_TEMPLATES = {
        'BiGG_Models': {
            'base_url': 'http://bigg.ucsd.edu/static/models/{model_id}.xml',
            'alternative_formats': ['.xml.gz'],
        },
        'VMH': {
            'base_url': 'https://www.vmh.life/files/models/{model_id}.xml',
            'alternative_formats': [],
        },
        'ModelSEED': {
            'base_url': 'https://modelseed.org/models/{model_id}/sbml',
            'alternative_formats': [],
        }
    }
    
    @staticmethod
    def verify_url_with_http(url: str, timeout: int = 10) -> Tuple[bool, str]:
        """
        HTTP HEADリクエストでURLの有効性を検証
        
        Args:
            url: 検証するURL
            timeout: タイムアウト秒数
        
        Returns:
            (is_valid, message)
        """
        try:
            response = requests.head(
                url, 
                timeout=timeout, 
                allow_redirects=True,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '').lower()
                valid_types = ['application/xml', 'text/xml', 'application/octet-stream', 'application/gzip']
                
                if any(vt in content_type for vt in valid_types):
                    return True, f"OK (Content-Type: {content_type})"
                else:
                    return False, f"無効なContent-Type: {content_type}"
            else:
                return False, f"HTTPステータス: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return False, "タイムアウト"
        except requests.exceptions.RequestException as e:
            return False, f"リクエストエラー: {str(e)}"
    
    @staticmethod
    def validate_downloaded_file(file_path: Path, min_size: int = 1024) -> Tuple[bool, str]:
        """
        ダウンロードされたファイルの妥当性を検証
        
        Args:
            file_path: ファイルパス
            min_size: 最小ファイルサイズ（バイト）
        
        Returns:
            (is_valid, message)
        """
        if not file_path.exists():
            return False, "ファイルが存在しません"
        
        file_size = file_path.stat().st_size
        if file_size < min_size:
            return False, f"ファイルサイズが小さすぎます: {file_size} bytes"
        
        # XMLパース可能性の確認
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            return True, f"有効なXML ({file_size:,} bytes)"
        except ET.ParseError as e:
            return False, f"XMLパースエラー: {str(e)}"
        except Exception as e:
            return False, f"検証エラー: {str(e)}"
    
    def _select_gemini_3_model(self, prefer_pro: bool = False) -> str:
        """
        Gemini 3系モデルを選択
        
        Args:
            prefer_pro: Proモデルを優先するか
        
        Returns:
            選択されたモデル名
        """
        try:
            print("🔍 Gemini 3系モデルを検索中...")
            
            # 利用可能なモデルのリストを取得
            models_response = self.client.models.list()
            
            # モデル名のリストを抽出
            available_models = []
            for model in models_response:
                model_name = model.name
                # models/ プレフィックスを除去
                if model_name.startswith('models/'):
                    model_name = model_name[7:]
                available_models.append(model_name)
            
            # Gemini 3系モデルのみをフィルタ
            gemini_3_models = [m for m in available_models if 'gemini-3' in m.lower()]
            
            if not gemini_3_models:
                print("⚠️  Gemini 3系モデルが見つかりません。利用可能なモデルから選択します。")
                gemini_3_models = available_models
            
            # 優先順位リスト
            if prefer_pro:
                priority_models = [
                    'gemini-3-pro',
                    'gemini-3.1-pro',
                    'gemini-3-flash',
                    'gemini-3.1-flash',
                    'gemini-3.1-flash-lite',
                ]
            else:
                priority_models = [
                    'gemini-3.1-flash-lite',
                    'gemini-3-flash',
                    'gemini-3.1-flash',
                    'gemini-3-pro',
                    'gemini-3.1-pro',
                ]
            
            # 優先順位に従ってモデルを選択
            for preferred in priority_models:
                for available in gemini_3_models:
                    if preferred in available.lower():
                        print(f"✅ 選択されたモデル: {available}")
                        return available
            
            # フォールバック: 最初のGemini 3系モデル
            if gemini_3_models:
                selected = gemini_3_models[0]
                print(f"⚠️  デフォルトGemini 3系モデルを使用: {selected}")
                return selected
            
            # 最終フォールバック
            fallback = 'gemini-1.5-flash-002'
            print(f"⚠️  Gemini 3系が利用不可、フォールバック: {fallback}")
            return fallback
            
        except Exception as e:
            print(f"⚠️  モデル選択エラー: {str(e)}")
            fallback = 'gemini-1.5-flash-002'
            print(f"📌 フォールバック: {fallback} を使用")
            return fallback
    
    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=self.api_key)
        
        # Gemini 3系モデルを優先的に選択
        self.model_name = self._select_gemini_3_model()
        self.model_name_pro = self._select_gemini_3_model(prefer_pro=True)
        
        # 出力ディレクトリの作成
        self.output_dir = Path('models/sbml')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path('gemini_outputs')
        self.log_dir.mkdir(exist_ok=True)
    
    def _call_gemini(
        self, 
        prompt: str, 
        task_name: str = "sbml_search",
        use_pro: bool = False,
        max_retries: int = 3
    ) -> str:
        """
        Gemini APIを呼び出す（リトライ機能付き）
        
        Args:
            prompt: プロンプト
            task_name: タスク名
            use_pro: 高性能モデル（Pro）を使用するか
            max_retries: 最大リトライ回数
        
        Returns:
            レスポンステキスト
        """
        model = self.model_name_pro if use_pro else self.model_name
        print(f"🤖 Gemini呼び出し中: {task_name} (モデル: {model})")
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = min(2 ** attempt, 16)  # 最大16秒
                    print(f"  ⏳ {wait_time}秒待機後に再試行... (試行 {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt
                )
                
                # ログ保存
                timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
                log_file = self.log_dir / f"{task_name}_{timestamp}.json"
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'task': task_name,
                        'timestamp': timestamp,
                        'model': model,
                        'attempt': attempt + 1,
                        'prompt': prompt,
                        'response': response.text
                    }, f, ensure_ascii=False, indent=2)
                
                return response.text
                
            except Exception as e:
                error_msg = str(e)
                
                # 503エラーの場合はリトライ
                if '503' in error_msg or 'UNAVAILABLE' in error_msg:
                    print(f"  ⚠️  503エラー (試行 {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        print(f"  ❌ リトライ上限に達しました")
                        return ""
                else:
                    print(f"  ❌ Geminiエラー: {e}")
                    return ""
        
        return ""
    
    def step1_select_organisms(self) -> Dict[str, str]:
        """
        ステップ1: サブエージェントによる微生物選定（複数候補）
        
        Returns:
            {species_name: rationale}
        """
        print("\n" + "="*60)
        print("📋 ステップ1: 微生物種の候補選定（各役割5種ずつ）")
        print("="*60)
        
        prompt = """# Task: 天然ゴム分解コンソーシアムの微生物候補選定

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
以下のJSON形式で**各役割につき5種の候補**を選定してください（合計15種）：

```json
{
  "lcp_degraders": [
    {
      "species_name": "Gordonia polyisoprenivorans",
      "strain": "VH2",
      "phylum": "Actinobacteria",
      "rationale": "Lcp1/Lcp2酵素を保有し、天然ゴムを効率的に低分子イソプレノイドに分解。",
      "key_capabilities": ["天然ゴム分解", "イソプレノイド生成"],
      "expected_auxotrophy": "アルギニン",
      "gem_availability_estimate": "文献 (PMID: 35149306)",
      "gem_quality_estimate": "高",
      "priority": 1
    },
    {
      "species_name": "Rhodococcus erythropolis",
      "strain": "PR4",
      "phylum": "Actinobacteria",
      "rationale": "Lcp相同酵素を保有、ゴム分解能あり",
      "key_capabilities": ["ゴム分解", "多様な炭化水素分解"],
      "expected_auxotrophy": "トリプトファン",
      "gem_availability_estimate": "BiGG Models / ModelSEED",
      "gem_quality_estimate": "中〜高",
      "priority": 2
    }
  ],
  "pha_accumulators": [
    {
      "species_name": "Cupriavidus necator",
      "strain": "H16",
      "phylum": "Proteobacteria",
      "rationale": "PHA生産のモデル株、高効率PHA蓄積",
      "key_capabilities": ["PHA合成", "炭素源多様性"],
      "expected_auxotrophy": "トリプトファン",
      "gem_availability_estimate": "BiGG Models (iJN1463)",
      "gem_quality_estimate": "非常に高",
      "priority": 1
    },
    {
      "species_name": "Pseudomonas putida",
      "strain": "KT2440",
      "phylum": "Proteobacteria",
      "rationale": "PHA生産能あり、多様な代謝能",
      "key_capabilities": ["PHA合成", "芳香族化合物分解"],
      "expected_auxotrophy": "ロイシン",
      "gem_availability_estimate": "BiGG Models (iJN1462)",
      "gem_quality_estimate": "非常に高",
      "priority": 2
    }
  ],
  "stabilizers": [
    {
      "species_name": "Pseudomonas fluorescens",
      "strain": "Pf-5",
      "phylum": "Proteobacteria",
      "rationale": "多様な有機酸代謝、pH調整、バイオフィルム形成",
      "key_capabilities": ["副産物代謝", "pH調整", "バイオフィルム"],
      "expected_auxotrophy": "メチオニン",
      "gem_availability_estimate": "ModelSEED / 文献",
      "gem_quality_estimate": "中〜高",
      "priority": 1
    },
    {
      "species_name": "Bacillus subtilis",
      "strain": "168",
      "phylum": "Firmicutes",
      "rationale": "pH調整、バイオフィルム形成、系の安定化",
      "key_capabilities": ["pH調整", "バイオフィルム", "胞子形成"],
      "expected_auxotrophy": "ヒスチジン",
      "gem_availability_estimate": "BiGG Models / ModelSEED",
      "gem_quality_estimate": "高",
      "priority": 2
    }
  ]
}
```

**重要**: 各役割につき最低5種、できれば7-10種の候補を提案してください。
GEM入手可能性が高い種を優先し、priorityフィールドで優先順位を明記してください。

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
                
                print("\n✅ 候補微生物:")
                
                # 全候補を統合
                all_candidates = {}
                
                for role, role_key in [
                    ('LCP分解菌', 'lcp_degraders'),
                    ('PHA蓄積菌', 'pha_accumulators'),
                    ('安定化菌', 'stabilizers')
                ]:
                    candidates = selection_data.get(role_key, [])
                    print(f"\n  【{role}】 {len(candidates)}種")
                    for org in candidates:
                        species = org['species_name']
                        print(f"    - {species} (優先度: {org.get('priority', 'N/A')})")
                        print(f"      GEM: {org.get('gem_availability_estimate', '不明')}")
                        
                        # 役割情報を追加
                        org['consortium_role'] = role
                        all_candidates[species] = org
                
                # 選定結果を保存
                selection_file = self.log_dir / 'organism_candidates.json'
                with open(selection_file, 'w', encoding='utf-8') as f:
                    json.dump(selection_data, f, ensure_ascii=False, indent=2)
                
                print(f"\n📊 合計候補数: {len(all_candidates)}種")
                
                return all_candidates
            else:
                print("⚠️  JSON形式のレスポンスが見つかりませんでした")
                return {}
                
        except Exception as e:
            print(f"❌ JSON解析エラー: {e}")
            return {}
    
    def _verify_and_fix_url(self, species_name: str, model_info: Dict) -> Dict:
        """
        URLを検証し、必要に応じて修正する（実際のHTTP検証使用）
        
        Args:
            species_name: 種名
            model_info: モデル情報
        
        Returns:
            修正されたモデル情報
        """
        url = model_info.get('download_url', '')
        model_id = model_info.get('model_id', '')
        
        print(f"  🔍 URL検証中: {url[:80]}...")
        
        # まず実際のHTTPリクエストで検証
        is_valid, message = self.verify_url_with_http(url)
        
        if is_valid:
            print(f"  ✅ URL検証成功: {message}")
            model_info['url_verified'] = True
            return model_info
        else:
            print(f"  ⚠️  URL検証失敗: {message}")
            
            # フォールバック: 既知のデータベースURLテンプレートを試行
            if model_id:
                print(f"  🔄 既知のデータベースURLテンプレートを試行中...")
                
                for db_name, template_info in self.DATABASE_URL_TEMPLATES.items():
                    base_url = template_info['base_url'].format(model_id=model_id)
                    
                    # 基本URLを試行
                    is_valid, msg = self.verify_url_with_http(base_url)
                    if is_valid:
                        print(f"  ✅ {db_name}で有効なURLを発見: {base_url}")
                        model_info['download_url'] = base_url
                        model_info['url_verified'] = True
                        model_info['source'] = f"{db_name} (自動検出)"
                        return model_info
                    
                    # 代替フォーマットを試行
                    for alt_format in template_info['alternative_formats']:
                        alt_url = base_url + alt_format
                        is_valid, msg = self.verify_url_with_http(alt_url)
                        if is_valid:
                            print(f"  ✅ {db_name}で有効なURLを発見: {alt_url}")
                            model_info['download_url'] = alt_url
                            model_info['url_verified'] = True
                            model_info['source'] = f"{db_name} (自動検出, {alt_format})"
                            return model_info
                
                print(f"  ❌ 既知のデータベースでも有効なURLが見つかりませんでした")
        
        # 最終手段: Geminiに問い合わせ
        print(f"  🤖 Geminiに代替URL検索を依頼中...")
        
        verify_prompt = f"""# Task: SBML モデルダウンロードURLの検証と修正

## 対象
- 種名: {species_name}
- モデルID: {model_info.get('model_id', '不明')}
- ソース: {model_info.get('source', '不明')}
- 提案URL: {url}

## タスク
提案されたURLが実際にアクセス可能か検証し、必要に応じて正しいURLを提案してください。

### 検証ポイント
1. **BiGG Models**: `http://bigg.ucsd.edu/static/models/[model_id].xml` 形式が正しいか
2. **GitHub**: `raw.githubusercontent.com` を使用しているか（`blob`ではなく）
3. **文献補足資料**: DOIから実際のファイルURLにリダイレクトされるか
4. **ModelSEED/KBase**: 正しいAPI エンドポイントを使用しているか

### 出力形式
```json
{{
  "url_valid": true,
  "corrected_url": "https://correct-url.com/file.xml",
  "access_method": "direct_download",
  "notes": "URLは正しい / GitHubのrawリンクに修正 / 等"
}}
```

**重要**: 実際に存在する、ダウンロード可能なURLのみを提案してください。
推測ではなく、データベースの実際の構造に基づいて回答してください。"""

        response = self._call_gemini(verify_prompt, f"url_verify_{species_name.replace(' ', '_')}", use_pro=True)
        
        if response:
            # JSON抽出
            try:
                json_start = response.find('```json')
                if json_start != -1:
                    json_start = response.find('\n', json_start) + 1
                    json_end = response.find('```', json_start)
                    json_str = response[json_start:json_end].strip()
                    verify_data = json.loads(json_str)
                    
                    corrected_url = verify_data.get('corrected_url', '')
                    if corrected_url:
                        # Geminiが提案したURLも実際に検証
                        is_valid, msg = self.verify_url_with_http(corrected_url)
                        if is_valid:
                            print(f"  ✅ Gemini提案URLが有効: {corrected_url[:80]}...")
                            model_info['download_url'] = corrected_url
                            model_info['url_verified'] = True
                            return model_info
                        else:
                            print(f"  ⚠️  Gemini提案URLも無効: {msg}")
                        
            except Exception as e:
                print(f"  ⚠️  Gemini応答の解析エラー: {e}")
        
        model_info['url_verified'] = False
        return model_info
    
    def step2_search_sbml_models(self, organisms: Dict[str, str]) -> Dict[str, List[Dict]]:
        """
        ステップ2: SBMLモデルの検索（URL検証付き）
        
        Args:
            organisms: 選定された微生物の辞書
        
        Returns:
            {species_name: [model_info]}
        """
        print("\n" + "="*60)
        print("🔍 ステップ2: SBMLモデルの検索とURL検証")
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

            response = self._call_gemini(
                prompt, 
                f"sbml_search_{species_name.replace(' ', '_')}",
                use_pro=True  # 検索には高性能モデルを使用
            )
            
            if not response:
                print(f"  ❌ 検索失敗（レスポンスなし）")
                all_models[species_name] = []
                continue
            
            # JSONを抽出
            try:
                json_start = response.find('```json')
                if json_start != -1:
                    json_start = response.find('\n', json_start) + 1
                    json_end = response.find('```', json_start)
                    json_str = response[json_start:json_end].strip()
                    search_data = json.loads(json_str)
                    
                    models = search_data.get('models_found', [])
                    
                    # 各モデルのURLを検証
                    verified_models = []
                    for model in models:
                        verified_model = self._verify_and_fix_url(species_name, model)
                        verified_models.append(verified_model)
                    
                    all_models[species_name] = verified_models
                    
                    print(f"  ✅ {len(verified_models)}件のモデルが見つかりました")
                    for model in verified_models:
                        verified_mark = "✓" if model.get('url_verified') else "?"
                        print(f"    - {model['model_id']} ({model['source']}) [{verified_mark}]")
                        print(f"      品質: {model['quality_score']}/10")
                
            except Exception as e:
                print(f"  ❌ JSON解析エラー: {e}")
                all_models[species_name] = []
        
        # 検索結果を保存
        search_file = self.log_dir / 'sbml_search_results.json'
        with open(search_file, 'w', encoding='utf-8') as f:
            json.dump(all_models, f, ensure_ascii=False, indent=2)
        
        return all_models
    
    def step2_5_filter_and_select_best(
        self, 
        organisms: Dict[str, str], 
        models_info: Dict[str, List[Dict]]
    ) -> Dict[str, str]:
        """
        ステップ2.5: モデル情報に基づいて最適な3種を選定
        
        Args:
            organisms: 候補微生物の辞書
            models_info: モデル情報の辞書
        
        Returns:
            最終選定された3種の辞書
        """
        print("\n" + "="*60)
        print("🔍 ステップ2.5: モデル評価と最適3種の選定")
        print("="*60)
        
        # 役割ごとに候補を分類
        role_candidates = {
            'LCP分解菌': [],
            'PHA蓄積菌': [],
            '安定化菌': []
        }
        
        for species_name, org_info in organisms.items():
            models = models_info.get(species_name, [])
            role = org_info.get('consortium_role', '')
            
            if not models:
                print(f"⚠️  {species_name}: モデルなし → スキップ")
                continue
            
            # 最高品質のモデルを取得
            best_model = max(models, key=lambda m: m.get('quality_score', 0))
            quality = best_model.get('quality_score', 0)
            
            if quality >= 6 and role in role_candidates:
                score = quality + org_info.get('priority', 5) * 0.5
                role_candidates[role].append({
                    'species_name': species_name,
                    'org_info': org_info,
                    'best_model': best_model,
                    'quality': quality,
                    'score': score
                })
                print(f"✅ {species_name} ({role}): 品質={quality}/10, スコア={score:.1f}")
        
        # 各役割から最高スコアの1種を選定
        final_selection = {}
        
        print("\n📊 最終選定:")
        for role, candidates in role_candidates.items():
            if not candidates:
                print(f"  ❌ {role}: 候補なし")
                continue
            
            # スコアでソート
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best = candidates[0]
            
            species_name = best['species_name']
            final_selection[species_name] = best['org_info']
            
            print(f"  ✅ {role}: {species_name}")
            print(f"     モデル: {best['best_model']['model_id']}")
            print(f"     品質: {best['quality']}/10")
            print(f"     スコア: {best['score']:.1f}")
            
            # 次点候補も表示
            if len(candidates) > 1:
                print(f"     次点: {candidates[1]['species_name']} (スコア: {candidates[1]['score']:.1f})")
        
        # 3種揃っているか確認
        if len(final_selection) < 3:
            print(f"\n⚠️  最終選定数が不足: {len(final_selection)}/3")
            missing_roles = [r for r in role_candidates.keys() if not any(
                org.get('consortium_role') == r for org in final_selection.values()
            )]
            print(f"  不足している役割: {', '.join(missing_roles)}")
        else:
            print(f"\n✅ 3種の選定完了")
        
        # 選定結果を保存
        selection_file = self.log_dir / 'final_selection.json'
        with open(selection_file, 'w', encoding='utf-8') as f:
            json.dump({
                'final_organisms': final_selection,
                'all_candidates': {
                    role: [c['species_name'] for c in candidates]
                    for role, candidates in role_candidates.items()
                }
            }, f, ensure_ascii=False, indent=2)
        
        return final_selection
    
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
                # ダウンロード試行（リトライ付き）
                max_download_retries = 3
                download_success = False
                
                for retry in range(max_download_retries):
                    try:
                        if retry > 0:
                            print(f"  🔄 ダウンロード再試行 ({retry + 1}/{max_download_retries})...")
                            time.sleep(2 ** retry)
                        
                        response = requests.get(
                            best_model['download_url'], 
                            timeout=30,
                            headers={'User-Agent': 'Mozilla/5.0'}
                        )
                        response.raise_for_status()
                        
                        # ファイル保存
                        file_path = self.output_dir / f"{safe_name}_{best_model['model_id']}.xml"
                        
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        
                        # ダウンロード後の検証
                        is_valid, validation_msg = self.validate_downloaded_file(file_path)
                        if is_valid:
                            print(f"  ✅ ダウンロード成功: {file_path}")
                            print(f"  ✅ ファイル検証: {validation_msg}")
                            downloaded_files[species_name] = file_path
                            download_success = True
                            break
                        else:
                            print(f"  ⚠️  ファイル検証失敗: {validation_msg}")
                            file_path.unlink()  # 無効なファイルを削除
                            if retry < max_download_retries - 1:
                                continue
                            else:
                                raise ValueError(f"ファイル検証失敗: {validation_msg}")
                        
                    except requests.exceptions.RequestException as e:
                        if retry < max_download_retries - 1:
                            print(f"  ⚠️  ダウンロードエラー: {e}")
                            continue
                        else:
                            raise
                
                if download_success:
                    continue
                    
            except Exception as e:
                print(f"  ❌ ダウンロード失敗: {e}")
                
                # Geminiに代替案を求める（高性能モデル使用）
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
                
                alt_response = self._call_gemini(
                    alt_prompt, 
                    f"alternative_download_{safe_name}",
                    use_pro=True  # 代替案検索には高性能モデルを使用
                )
                
                if alt_response:
                    print(f"  📝 代替案受信")
                    
                    # 代替URLを試行
                    try:
                        json_start = alt_response.find('```json')
                        if json_start != -1:
                            json_start = alt_response.find('\n', json_start) + 1
                            json_end = alt_response.find('```', json_start)
                            json_str = alt_response[json_start:json_end].strip()
                            alt_data = json.loads(json_str)
                            
                            alt_urls = alt_data.get('alternative_urls', [])
                            for alt_url in alt_urls:
                                print(f"  🔄 代替URL試行: {alt_url[:80]}...")
                                try:
                                    response = requests.get(alt_url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
                                    response.raise_for_status()
                                    
                                    file_path = self.output_dir / f"{safe_name}_{best_model['model_id']}.xml"
                                    with open(file_path, 'wb') as f:
                                        f.write(response.content)
                                    
                                    print(f"  ✅ 代替URLでダウンロード成功: {file_path}")
                                    downloaded_files[species_name] = file_path
                                    break
                                    
                                except Exception as e2:
                                    print(f"  ❌ 代替URLも失敗: {e2}")
                                    continue
                    except Exception as e3:
                        print(f"  ⚠️  代替案解析エラー: {e3}")
                else:
                    print(f"  ❌ 代替案取得失敗")
                
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
                
                # COBRApyで基本統計を計算
                try:
                    import cobra
                    model = cobra.io.read_sbml_model(str(file_path))
                    
                    basic_stats = {
                        'genes': len(model.genes),
                        'reactions': len(model.reactions),
                        'metabolites': len(model.metabolites),
                        'compartments': len(model.compartments)
                    }
                    
                    print(f"  📊 COBRApy統計:")
                    print(f"    遺伝子: {basic_stats['genes']}")
                    print(f"    反応: {basic_stats['reactions']}")
                    print(f"    代謝物: {basic_stats['metabolites']}")
                    
                except Exception as e:
                    print(f"  ⚠️  COBRApy解析エラー: {e}")
                    basic_stats = None
                
                # Geminiに詳細評価を依頼（軽量化）
                with open(file_path, 'r', encoding='utf-8') as f:
                    sbml_content = f.read()
                
                # ファイルサイズに応じて切り詰め
                max_chars = 50000
                if len(sbml_content) > max_chars:
                    # 先頭1000行を抽出
                    lines = sbml_content.split('\n')
                    sbml_sample = '\n'.join(lines[:1000]) + f"\n... (残り{len(lines)-1000}行を省略)"
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
                
                # リトライ戦略を強化（指数バックオフ: 2s, 4s, 8s, 16s）
                eval_response = self._call_gemini(
                    eval_prompt, 
                    f"evaluation_{species_name.replace(' ', '_')}",
                    use_pro=False,
                    max_retries=4  # リトライ回数を増やす
                )
                
                if not eval_response:
                    print(f"  ⚠️  Gemini評価失敗、簡易評価モードにフォールバック")
                    # 簡易評価: COBRApyの統計のみ
                    if basic_stats:
                        evaluations[species_name] = {
                            'status': 'simple_evaluation',
                            'file_path': str(file_path),
                            'statistics': basic_stats,
                            'quality': {
                                'quality_score': 6,  # デフォルトスコア
                                'evaluation_method': 'COBRApy only (Gemini unavailable)'
                            },
                            'basic_stats': stats
                        }
                        print(f"  ✅ 簡易評価完了（COBRApyのみ）")
                    else:
                        evaluations[species_name] = {
                            'status': 'partial',
                            'file_path': str(file_path),
                            'basic_stats': stats
                        }
                    continue
                
                # JSON抽出
                try:
                    json_start = eval_response.find('```json')
                    if json_start != -1:
                        json_start = eval_response.find('\n', json_start) + 1
                        json_end = eval_response.find('```', json_start)
                        json_str = eval_response[json_start:json_end].strip()
                        eval_data = json.loads(json_str)
                        
                        # COBRApyの統計があれば上書き
                        if basic_stats:
                            eval_data['statistics'] = basic_stats
                        
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
        
        # ステップ2.5: 最適な3種を選定
        final_organisms = self.step2_5_filter_and_select_best(organisms, models_info)
        
        if len(final_organisms) < 3:
            print(f"\n⚠️  最終選定数が不足: {len(final_organisms)}/3")
            print(f"  → パイプラインを中断します")
            return
        
        # 最終選定されたモデル情報のみを使用
        filtered_models_info = {k: v for k, v in models_info.items() if k in final_organisms}
        
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
