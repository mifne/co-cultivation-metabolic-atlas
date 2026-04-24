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
    species_names = ['Gordonia_polyisoprenivorans', 'Cupriavidus_necator', 'Pseudomonas_putida']
    
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
    
    for sbml_file in sbml_files:
        species_name = sbml_file.stem
        print(f"📂 読み込み中: {sbml_file.name}")
        model = cobra.io.read_sbml_model(str(sbml_file))
        models[species_name] = model
    
    return models


def setup_simulator(models: dict, use_mock: bool = True) -> dFBASimulator:
    """
    dFBAシミュレーターをセットアップ
    
    Args:
        models: COBRAモデル辞書
        use_mock: モックデータを使用するか
    
    Returns:
        dFBASimulator
    """
    # 初期条件（モデルの種名に基づいて動的に設定）
    initial_biomass = {}
    for species_name in models.keys():
        initial_biomass[species_name] = 0.1
    
    initial_metabolites = {
        'glc_e': 10.0,  # mM
        'arg_e': 0.5,
        'trp_e': 0.5,
        'leu_e': 0.5,
        'isoprene': 0.0
    }
    
    initial_rubber = 10.0  # g/L
    
    simulator = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=initial_rubber,
        volume=1.0,
        dt=0.1
    )
    
    return simulator


def train_agent(args):
    """エージェントを訓練"""
    print("=" * 60)
    print("🧬 dFBA-RL コンソーシアム制御システム - 訓練モード")
    print("=" * 60)
    
    # モデルの読み込み
    if args.sbml_dir:
        models = load_sbml_models(Path(args.sbml_dir))
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
    
    # PPOエージェントの作成
    agent = ConsortiumPPOAgent(
        env=env,
        learning_rate=args.learning_rate,
        verbose=1
    )
    
    # 訓練
    history = agent.train(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval
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
    
    # 結果の保存
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📈 評価結果保存: {results_path}")
    
    print("=" * 60)
    print("✅ 訓練完了")
    print("=" * 60)


def evaluate_agent(args):
    """訓練済みエージェントを評価"""
    print("=" * 60)
    print("📊 dFBA-RL コンソーシアム制御システム - 評価モード")
    print("=" * 60)
    
    # モデルの読み込み
    if args.sbml_dir:
        models = load_sbml_models(Path(args.sbml_dir))
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
    
    print("=" * 60)
    print("✅ 評価完了")
    print("=" * 60)


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
    train_parser.add_argument('--eval-episodes', type=int, default=10,
                             help='評価エピソード数')
    
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
