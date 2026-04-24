"""
Main Script for dFBA-RL Consortium Control
天然ゴム分解コンソーシアムのdFBA-RL制御メインスクリプト
"""

import argparse
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import cobra

from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.ppo_agent import ConsortiumPPOAgent


def create_mock_models() -> dict:
    """
    モックのCOBRAモデルを作成（実際のSBMLファイルがない場合のテスト用）
    
    Returns:
        {species_name: cobra.Model}
    """
    print("⚠️  モックモデルを使用（実際のSBMLファイルを使用する場合は--sbml-dirを指定）")
    
    models = {}
    species_names = ['Sphingobium_japonicum', 'Pseudomonas_putida_KT2440', 'Lactobacillus_plantarum']
    
    for species in species_names:
        # 簡単なモックモデルを作成
        model = cobra.Model(f'{species}_mock')
        
        # 代謝物
        glc = cobra.Metabolite('glc_e', compartment='e', name='Glucose')
        arg = cobra.Metabolite('arg_e', compartment='e', name='Arginine')
        trp = cobra.Metabolite('trp_e', compartment='e', name='Tryptophan')
        leu = cobra.Metabolite('leu_e', compartment='e', name='Leucine')
        biomass = cobra.Metabolite('biomass', compartment='c', name='Biomass')
        
        # 交換反応
        ex_glc = cobra.Reaction('EX_glc_e')
        ex_glc.add_metabolites({glc: -1})
        ex_glc.bounds = (-10, 0)
        
        ex_arg = cobra.Reaction('EX_arg_e')
        ex_arg.add_metabolites({arg: -1})
        ex_arg.bounds = (-10, 0)
        
        ex_trp = cobra.Reaction('EX_trp_e')
        ex_trp.add_metabolites({trp: -1})
        ex_trp.bounds = (-10, 0)
        
        ex_leu = cobra.Reaction('EX_leu_e')
        ex_leu.add_metabolites({leu: -1})
        ex_leu.bounds = (-10, 0)
        
        # バイオマス反応
        biomass_rxn = cobra.Reaction('BIOMASS')
        biomass_rxn.add_metabolites({
            glc: -1,
            arg: -0.1,
            trp: -0.05,
            leu: -0.08,
            biomass: 1
        })
        biomass_rxn.bounds = (0, 1000)
        
        # モデルに追加
        model.add_reactions([ex_glc, ex_arg, ex_trp, ex_leu, biomass_rxn])
        model.objective = 'BIOMASS'
        
        models[species] = model
    
    return models


def load_sbml_models(sbml_dir: Path) -> dict:
    """
    SBMLファイルからモデルを読み込み
    
    Args:
        sbml_dir: SBMLファイルのディレクトリ
    
    Returns:
        {species_name: cobra.Model}
    """
    models = {}
    sbml_files = list(sbml_dir.glob('*.xml')) + list(sbml_dir.glob('*.sbml'))
    
    if not sbml_files:
        raise FileNotFoundError(f"SBMLファイルが見つかりません: {sbml_dir}")
    
    print(f"\n📁 SBMLファイル検出: {len(sbml_files)}件")
    
    for sbml_file in sbml_files:
        species_name = sbml_file.stem
        
        # 空ファイルをスキップ
        file_size = sbml_file.stat().st_size
        if file_size == 0:
            print(f"⚠️  スキップ（空ファイル）: {sbml_file.name}")
            continue
        
        if file_size < 1024:  # 1KB未満
            print(f"⚠️  スキップ（ファイルサイズが小さすぎる: {file_size} bytes）: {sbml_file.name}")
            continue
        
        print(f"📂 読み込み中: {sbml_file.name} ({file_size:,} bytes)")
        
        try:
            model = cobra.io.read_sbml_model(str(sbml_file))
            
            # 基本的な検証
            if len(model.genes) == 0:
                print(f"  ⚠️  警告: 遺伝子が0個です")
            if len(model.reactions) == 0:
                print(f"  ⚠️  警告: 反応が0個です")
                continue
            
            models[species_name] = model
            print(f"  ✅ 成功: {len(model.genes)} genes, {len(model.reactions)} reactions")
            
        except Exception as e:
            print(f"  ❌ エラー: {sbml_file.name}")
            print(f"     {type(e).__name__}: {str(e)}")
            print(f"  → このファイルをスキップします")
            continue
    
    if not models:
        raise ValueError(f"有効なSBMLモデルが1つも読み込めませんでした: {sbml_dir}")
    
    print(f"\n✅ 読み込み成功: {len(models)}種のモデル")
    
    return models


def select_consortium_models(models: dict) -> dict:
    """
    読み込まれたモデルから最適な3種を選定
    
    Args:
        models: 全モデルの辞書
    
    Returns:
        選定された3種のモデル辞書
    """
    # 優先順位リスト（ドキュメントに基づく）
    priority_species = [
        # LCP分解菌
        ('Sphingobium_japonicum_iJN1463', 'LCP分解菌'),
        ('Sphingobium_japonicum', 'LCP分解菌'),
        
        # PHA蓄積菌
        ('Pseudomonas_putida_KT2440', 'PHA蓄積菌'),
        ('Pseudomonas_putida_KT2440_iJN1462', 'PHA蓄積菌'),
        ('Escherichia_coli_K12_iML1515', 'PHA蓄積菌（代替）'),
        
        # 安定化菌
        ('Lactobacillus_plantarum_iNF517', '安定化菌'),
        ('Lactobacillus_plantarum', '安定化菌'),
        ('Bacillus_subtilis_168_iYO844', '安定化菌（代替）'),
    ]
    
    selected = {}
    selected_roles = set()
    
    print("\n🔍 コンソーシアム用の3種を選定中...")
    
    for species_key, role in priority_species:
        # 役割が既に選定済みならスキップ
        role_type = role.split('（')[0]  # "PHA蓄積菌（代替）" -> "PHA蓄積菌"
        if role_type in selected_roles:
            continue
        
        # モデルが存在するか確認
        if species_key in models:
            selected[species_key] = models[species_key]
            selected_roles.add(role_type)
            print(f"  ✅ {role}: {species_key}")
            
            if len(selected) == 3:
                break
    
    if len(selected) < 3:
        print(f"\n⚠️  優先リストから3種選定できませんでした（{len(selected)}/3）")
        print(f"  → 利用可能なモデルから補完します")
        
        # 不足分を補完
        for species_name, model in models.items():
            if species_name not in selected:
                selected[species_name] = model
                print(f"  ✅ 補完: {species_name}")
                
                if len(selected) == 3:
                    break
    
    if len(selected) < 3:
        raise ValueError(f"3種のモデルを選定できませんでした（{len(selected)}/3種のみ利用可能）")
    
    print(f"\n✅ 最終選定: {len(selected)}種")
    return selected


def setup_simulator(models: dict, use_mock: bool = True) -> dFBASimulator:
    """
    dFBAシミュレーターをセットアップ
    
    Args:
        models: COBRAモデル辞書（3種）
        use_mock: モックデータを使用するか
    
    Returns:
        dFBASimulator
    """
    # 初期バイオマス（種ごとに異なる初期値）
    initial_biomass = {}
    for species_name in models.keys():
        if 'Sphingobium' in species_name:
            # LCP分解菌: 少量から開始（ゴム分解が進むと増殖）
            initial_biomass[species_name] = 0.05
        elif 'Pseudomonas' in species_name:
            # PHA蓄積菌: 中程度の初期値
            initial_biomass[species_name] = 0.1
        elif 'Lactobacillus' in species_name:
            # 安定化菌: 多めに開始（pH調整のため）
            initial_biomass[species_name] = 0.15
        else:
            initial_biomass[species_name] = 0.1
    
    # 初期代謝物濃度（BiGG Modelsの標準IDを使用）
    initial_metabolites = {
        # 主要炭素源（BiGG Models標準ID）
        'glc__D_e': 20.0,      # D-グルコース [mM]
        
        # アミノ酸（BiGG Models標準ID）
        'arg__L_e': 2.0,       # L-アルギニン [mM]
        'trp__L_e': 1.0,       # L-トリプトファン [mM]
        'leu__L_e': 1.5,       # L-ロイシン [mM]
        
        # その他の必須アミノ酸（増殖に必要）
        'ala__L_e': 1.0,       # L-アラニン [mM]
        'asn__L_e': 1.0,       # L-アスパラギン [mM]
        'asp__L_e': 1.0,       # L-アスパラギン酸 [mM]
        'cys__L_e': 0.5,       # L-システイン [mM]
        'gln__L_e': 1.0,       # L-グルタミン [mM]
        'glu__L_e': 1.0,       # L-グルタミン酸 [mM]
        'gly_e': 1.0,          # グリシン [mM]
        'his__L_e': 0.5,       # L-ヒスチジン [mM]
        'ile__L_e': 1.0,       # L-イソロイシン [mM]
        'lys__L_e': 1.0,       # L-リジン [mM]
        'met__L_e': 0.5,       # L-メチオニン [mM]
        'phe__L_e': 0.8,       # L-フェニルアラニン [mM]
        'pro__L_e': 1.0,       # L-プロリン [mM]
        'ser__L_e': 1.0,       # L-セリン [mM]
        'thr__L_e': 1.0,       # L-トレオニン [mM]
        'tyr__L_e': 0.5,       # L-チロシン [mM]
        'val__L_e': 1.0,       # L-バリン [mM]
        
        # 窒素源
        'nh4_e': 20.0,         # アンモニウム [mM]
        
        # リン酸
        'pi_e': 10.0,          # リン酸 [mM]
        
        # 硫黄源
        'so4_e': 5.0,          # 硫酸 [mM]
        
        # 酸素（好気条件）
        'o2_e': 21.0,          # 酸素 [mM]
        
        # 微量元素
        'fe2_e': 0.01,         # 鉄(II) [mM]
        'fe3_e': 0.01,         # 鉄(III) [mM]
        'ca2_e': 0.5,          # カルシウム [mM]
        'cl_e': 1.0,           # 塩化物 [mM]
        'co2_e': 1.0,          # 二酸化炭素 [mM]
        'cu2_e': 0.001,        # 銅 [mM]
        'h_e': 0.0001,         # プロトン（pH 7相当）[mM]
        'h2o_e': 55000.0,      # 水 [mM]
        'k_e': 5.0,            # カリウム [mM]
        'mg2_e': 2.0,          # マグネシウム [mM]
        'mn2_e': 0.01,         # マンガン [mM]
        'mobd_e': 0.001,       # モリブデン酸 [mM]
        'na1_e': 10.0,         # ナトリウム [mM]
        'zn2_e': 0.01,         # 亜鉛 [mM]
        
        # ビタミン類
        'thm_e': 0.01,         # チアミン [mM]
        'ribflv_e': 0.01,      # リボフラビン [mM]
        
        # イソプレノイド（初期は0、ゴム分解で生成）
        'isoprene': 0.0,
        
        # 有機酸（初期は微量）
        'ac_e': 0.1,           # 酢酸 [mM]
        'lac__D_e': 0.0,       # D-乳酸 [mM]
        'lac__L_e': 0.0,       # L-乳酸 [mM]
    }
    
    # 天然ゴム初期濃度
    initial_rubber = 10.0  # g/L
    
    # タイムステップを長めに設定（FBAの安定性向上）
    dt = 0.5  # 0.5時間 = 30分
    
    simulator = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=initial_rubber,
        volume=1.0,
        dt=dt
    )
    
    print(f"\n  🧪 初期条件:")
    print(f"    バイオマス: {initial_biomass}")
    print(f"    グルコース: {initial_metabolites.get('glc__D_e', 0):.1f} mM")
    print(f"    アミノ酸: Arg={initial_metabolites.get('arg__L_e', 0):.1f}, "
          f"Trp={initial_metabolites.get('trp__L_e', 0):.1f}, "
          f"Leu={initial_metabolites.get('leu__L_e', 0):.1f} mM")
    print(f"    窒素源: NH4={initial_metabolites.get('nh4_e', 0):.1f} mM")
    print(f"    リン酸: Pi={initial_metabolites.get('pi_e', 0):.1f} mM")
    print(f"    酸素: O2={initial_metabolites.get('o2_e', 0):.1f} mM")
    print(f"    天然ゴム: {initial_rubber:.1f} g/L")
    print(f"    タイムステップ: {dt} h")
    print(f"    代謝物総数: {len(initial_metabolites)}種")
    
    return simulator


def train_agent(args):
    """エージェントを訓練"""
    print("🧬 訓練開始")
    
    # モデルの読み込み
    if hasattr(args, 'sbml_dir') and args.sbml_dir:
        all_models = load_sbml_models(Path(args.sbml_dir))
        models = select_consortium_models(all_models)
    else:
        models = create_mock_models()
    
    # シミュレーターのセットアップ
    simulator = setup_simulator(models, use_mock=(not hasattr(args, 'sbml_dir') or args.sbml_dir is None))
    
    # RL環境の作成
    env = ConsortiumEnv(
        simulator=simulator,
        max_steps=args.max_steps,
        target_rubber_degradation=args.target_degradation,
        amino_acid_cost=args.amino_acid_cost
    )
    
    # 環境のリセット動作を確認
    obs, info = env.reset()
    
    # 評価用環境の作成（オプション）
    eval_env = None
    if hasattr(args, 'eval_during_training') and args.eval_during_training:
        eval_simulator = setup_simulator(models, use_mock=(not hasattr(args, 'sbml_dir') or args.sbml_dir is None))
        eval_env = ConsortiumEnv(
            simulator=eval_simulator,
            max_steps=args.max_steps,
            target_rubber_degradation=args.target_degradation,
            amino_acid_cost=args.amino_acid_cost
        )
    
    # PPOエージェントの作成
    agent = ConsortiumPPOAgent(
        env=env,
        learning_rate=args.learning_rate,
        device='cpu',
        verbose=1,
        n_envs=1  # 並列環境は将来的に対応
    )
    
    # 訓練
    history = agent.train(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        save_freq=getattr(args, 'save_freq', 10000),
        save_path=args.output_dir,
        eval_freq=getattr(args, 'eval_freq', 5000) if getattr(args, 'eval_during_training', False) else None,
        eval_env=eval_env,
        n_eval_episodes=args.eval_episodes
    )
    
    # モデルの保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / 'ppo_consortium_model.zip'
    agent.save(str(model_path))
    
    # 訓練履歴の保存
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"📊 訓練履歴保存: {history_path}")
    
    # 評価
    results = agent.evaluate(n_episodes=args.eval_episodes)
    
    # 結果の保存（numpy型をPython標準型に変換）
    def convert_to_serializable(obj):
        """numpy型をJSON serializable型に変換"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"✅ 訓練完了: {results_path}")


def evaluate_agent(args):
    """訓練済みエージェントを評価"""
    print("📊 評価開始")
    
    # モデルの読み込み
    if args.sbml_dir:
        all_models = load_sbml_models(Path(args.sbml_dir))
        # 3種に絞り込み
        models = select_consortium_models(all_models)
    else:
        models = create_mock_models()
    
    # シミュレーターのセットアップ
    simulator = setup_simulator(models, use_mock=(args.sbml_dir is None))
    
    # RL環境の作成
    env = ConsortiumEnv(
        simulator=simulator,
        max_steps=args.max_steps,
        target_rubber_degradation=args.target_degradation,
        amino_acid_cost=args.amino_acid_cost
    )
    
    # エージェントの読み込み
    agent = ConsortiumPPOAgent(env=env)
    agent.load(args.model_path)
    
    # 評価
    results = agent.evaluate(n_episodes=args.eval_episodes)
    
    print("✅ 評価完了")


def main():
    parser = argparse.ArgumentParser(
        description='天然ゴム分解コンソーシアムのdFBA-RL制御システム'
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='実行モード')
    
    # 訓練モード
    train_parser = subparsers.add_parser('train', help='エージェントを訓練')
    train_parser.add_argument('--sbml-dir', type=str, help='SBMLファイルのディレクトリ')
    train_parser.add_argument('--total-timesteps', type=int, default=100000,
                             help='総訓練ステップ数')
    train_parser.add_argument('--max-steps', type=int, default=100,
                             help='1エピソードの最大ステップ数')
    train_parser.add_argument('--target-degradation', type=float, default=0.9,
                             help='目標ゴム分解率')
    train_parser.add_argument('--amino-acid-cost', type=float, default=0.1,
                             help='アミノ酸コスト係数')
    train_parser.add_argument('--learning-rate', type=float, default=3e-4,
                             help='学習率')
    train_parser.add_argument('--output-dir', type=str, default='outputs',
                             help='出力ディレクトリ')
    train_parser.add_argument('--log-interval', type=int, default=10,
                             help='ログ出力間隔')
    train_parser.add_argument('--save-freq', type=int, default=10000,
                             help='チェックポイント保存頻度')
    train_parser.add_argument('--eval-episodes', type=int, default=10,
                             help='評価エピソード数')
    train_parser.add_argument('--eval-during-training', action='store_true',
                             help='訓練中に定期的に評価を実行')
    train_parser.add_argument('--eval-freq', type=int, default=5000,
                             help='訓練中評価の頻度')
    
    # 評価モード
    eval_parser = subparsers.add_parser('evaluate', help='訓練済みエージェントを評価')
    eval_parser.add_argument('--model-path', type=str, required=True,
                            help='訓練済みモデルのパス')
    eval_parser.add_argument('--sbml-dir', type=str, help='SBMLファイルのディレクトリ')
    eval_parser.add_argument('--max-steps', type=int, default=100,
                            help='1エピソードの最大ステップ数')
    eval_parser.add_argument('--target-degradation', type=float, default=0.9,
                            help='目標ゴム分解率')
    eval_parser.add_argument('--amino-acid-cost', type=float, default=0.1,
                            help='アミノ酸コスト係数')
    eval_parser.add_argument('--eval-episodes', type=int, default=10,
                            help='評価エピソード数')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_agent(args)
    elif args.mode == 'evaluate':
        evaluate_agent(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
