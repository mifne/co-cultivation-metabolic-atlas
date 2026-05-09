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
from typing import Dict, Tuple, Optional, List, Callable

from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.ppo_agent import ConsortiumPPOAgent
from src.callbacks import ConsortiumCallback


def make_env(sbml_dir: str, env_params: dict, data_log_path: Optional[str] = None, rank: int = 0):
    """
    環境作成用のファクトリ関数を返す。
    """
    def _init():
        if sbml_dir:
            all_models = load_sbml_models(Path(sbml_dir))
            models = select_consortium_models(all_models)
        else:
            models = create_mock_models()
            
        initial_biomass, initial_metabolites = get_initial_params(models)
        
        # 並列環境ごとに異なるログファイル名を生成
        env_log_path = None
        if data_log_path:
            log_path_obj = Path(data_log_path)
            env_log_path = str(log_path_obj.parent / f"{log_path_obj.stem}_env{rank}{log_path_obj.suffix}")

        sim = dFBASimulator(
            models=models,
            initial_biomass=initial_biomass,
            initial_metabolites=initial_metabolites,
            initial_rubber=100.0,
            volume=1.0,
            dt=0.2,
            data_log_path=env_log_path
        )
        
        env = ConsortiumEnv(simulator=sim, **env_params)
        return env
    return _init


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
        nh4 = cobra.Metabolite('nh4_e', compartment='e', name='Ammonium')
        biomass = cobra.Metabolite('biomass', compartment='c', name='Biomass')
        
        # 交換反応
        ex_glc = cobra.Reaction('EX_glc_e')
        ex_glc.add_metabolites({glc: -1})
        ex_glc.bounds = (-100, 1000)
        
        ex_nh4 = cobra.Reaction('EX_nh4_e')
        ex_nh4.add_metabolites({nh4: -1})
        ex_nh4.bounds = (-100, 1000)
        
        ex_arg = cobra.Reaction('EX_arg_e')
        ex_arg.add_metabolites({arg: -1})
        ex_arg.bounds = (-10, 1000)
        
        ex_trp = cobra.Reaction('EX_trp_e')
        ex_trp.add_metabolites({trp: -1})
        ex_trp.bounds = (-10, 1000)
        
        ex_leu = cobra.Reaction('EX_leu_e')
        ex_leu.add_metabolites({leu: -1})
        ex_leu.bounds = (-10, 1000)
        
        # バイオマス反応
        biomass_rxn = cobra.Reaction('BIOMASS')
        biomass_rxn.add_metabolites({
            glc: -1,
            nh4: -0.5,
            arg: -0.1,
            trp: -0.05,
            leu: -0.08,
            biomass: 1
        })
        biomass_rxn.bounds = (0, 1000)
        
        # モデルに追加
        model.add_reactions([ex_glc, ex_nh4, ex_arg, ex_trp, ex_leu, biomass_rxn])
        model.objective = 'BIOMASS'
        
        models[species] = model
    
    return models


def load_sbml_models(sbml_dir: Path) -> dict:
    """
    SBMLファイルからモデルを読み込み
    """
    models = {}
    sbml_files = list(sbml_dir.glob('*.xml')) + list(sbml_dir.glob('*.sbml'))
    
    if not sbml_files:
        raise FileNotFoundError(f"SBMLファイルが見つかりません: {sbml_dir}")
    
    # print(f"\n📁 SBMLファイル検出: {len(sbml_files)}件") # Quiet
    
    for sbml_file in sbml_files:
        species_name = sbml_file.stem
        try:
            model = cobra.io.read_sbml_model(str(sbml_file))
            models[species_name] = model
        except Exception:
            continue
    
    if not models:
        raise ValueError(f"有効なSBMLモデルが1つも読み込めませんでした: {sbml_dir}")
    
    return models


def select_consortium_models(models: dict) -> dict:
    """
    真の精鋭3種（OR16, NS21, LP）を選定
    """
    priority_species = [
        ('Actinoplanes_sp_OR16_lcp', 'Engine 1: Lcp分解'),
        ('Rhizobacter_gummiphilus_NS21', 'Engine 2: Rox分解 + PHA蓄積'),
        ('Lactobacillus_plantarum', 'Stabilizer: 代謝安定化'),
    ]
    selected = {}
    for key, role in priority_species:
        if key in models:
            selected[key] = models[key]
            print(f"  ✅ {role}: {key}")
    if len(selected) < 3:
        for name, model in models.items():
            if name not in selected and len(selected) < 3:
                selected[name] = model
                print(f"  ✅ 補完: {name}")
    return selected



def get_initial_params(models: dict) -> Tuple[dict, dict]:
    """
    初期パラメータを取得（M9 + LP生存用サプリメント）
    科学的調整: OR16優位の初期比率と低糖条件により共生を誘導
    """
    # OR16を主役に、LPとNS21をサポーターとして1:5の比率で開始
    initial_biomass = {
        name: 0.5 if 'OR16' in name else 0.1 
        for name in models.keys()
    }
    
    initial_metabolites = {
        'glc__D_e': 0.1, 'nh4_e': 50.0, 'pi_e': 50.0, 'o2_e': 0.25,
        'so4_e': 2.0, 'mg2_e': 2.0, 'ca2_e': 0.1, 'k_e': 10.0, 'cl_e': 10.0,
        'fe3_e': 0.1, 'fe2_e': 0.1, 'h_e': 0.0001, 'h2o_e': 55000.0, 'co2_e': 1.0,
        'zn2_e': 0.01, 'mn2_e': 0.1, 'cu2_e': 0.01, 'cobalt2_e': 0.01,
        'ni2_e': 0.01, 'mobd_e': 0.01,
        # --- 必須ビタミン・補酵素 (LP要求分) ---
        'nac_e': 0.1, 'ribflv_e': 0.1, 'pnto__R_e': 0.1, 'thm_e': 0.1,
        'btn_e': 0.1, '4abz_e': 0.1, 'fol_e': 0.1, 'nicnt_e': 0.1,
        'ade_e': 0.1, 'gua_e': 0.1, 'ura_e': 0.1, 'xan_e': 0.1, 'orot_e': 0.1,
        'ins_e': 0.1, 'thymd_e': 0.1,
        # --- 実用化修正: 個別アミノ酸を廃止し、酵母エキスに集約 ---
        'yeast_extract_e': 1.0,

        # --- バイオサーファクタント代替 (2-methylbutanoic acid) ---
        '2mba_e': 0.0,
        # --- ゴム中間体 ---
        'C30_oligo_e': 0.0, 'odtd_e': 0.0,
        # --- 種特異的栄養素の初期値 ---
        'mlttr_e': 0.0, 'ptrc_e': 0.0, 'mnl_e': 0.0
    }

    return initial_biomass, initial_metabolites



def setup_simulator(models: dict, data_log_path: Optional[str] = None) -> dFBASimulator:
    """
    dFBAシミュレーターをセットアップ
    """
    initial_biomass, initial_metabolites = get_initial_params(models)
    initial_rubber = 100.0
    dt = 0.1
    
    simulator = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=initial_rubber,
        volume=1.0,
        dt=dt,
        data_log_path=data_log_path
    )
    return simulator


def convert_to_serializable(obj):
    """
    JSONシリアル化不可能なオブジェクトを変換
    """
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


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    線形学習率スケジュール
    :param initial_value: 初期の学習率
    :return: 残りのステップ数（1.0から0.0）に応じた学習率を返す関数
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def train_agent(args):
    """エージェントを訓練"""
    print("🧬 訓練開始")
    
    # モデルの読み込み
    if hasattr(args, 'sbml_dir') and args.sbml_dir:
        all_models = load_sbml_models(Path(args.sbml_dir))
        models = select_consortium_models(all_models)
    else:
        models = create_mock_models()
    
    # パラメータ設定
    initial_biomass, initial_metabolites = get_initial_params(models)
    simulator_params = {
        'models': models,
        'initial_biomass': initial_biomass,
        'initial_metabolites': initial_metabolites,
        'initial_rubber': 100.0,
        'volume': 1.0,
        'dt': 0.1,
        'data_log_path': getattr(args, 'data_log_path', None)
    }
    
    env_params = {
        'max_time': args.max_steps * 0.2, # dt=0.2に合わせて時間に変換
    }

    # 学習率スケジュールの設定
    lr = args.learning_rate
    if getattr(args, 'linear_lr', False):
        print(f"📉 線形学習率スケジュールを適用 (Initial LR: {lr})")
        lr = linear_schedule(lr)

    # 環境の作成
    if args.n_envs > 1:
        # SubprocVecEnv用の関数のリストを作成
        env_input = [make_env(args.sbml_dir, env_params, getattr(args, 'data_log_path', None), rank=i) for i in range(args.n_envs)]
    else:
        # 単一環境（DummyVecEnv用）
        env_input = make_env(args.sbml_dir, env_params, getattr(args, 'data_log_path', None), rank=0)

    # PPOエージェントの作成
    agent = ConsortiumPPOAgent(
        env=env_input,
        learning_rate=lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        device='cpu',
        verbose=1,
        n_envs=args.n_envs,
        tensorboard_log=args.tensorboard_log
    )

    # チェックポイントから再開
    if args.resume_from:
        print(f"🔄 チェックポイントから再開: {args.resume_from}")
        agent.load(args.resume_from)

    # 訓練
    from stable_baselines3.common.callbacks import CheckpointCallback
    
    callbacks = [
        ConsortiumCallback(),
        CheckpointCallback(
            save_freq=max(1000, args.save_freq // args.n_envs),
            save_path=args.output_dir,
            name_prefix='ppo_consortium',
            save_vecnormalize=True
        )
    ]
    
    print(f"🚀 学習開始: {args.total_timesteps} ステップ")
    print(f"📡 リアルタイム監視: TensorBoard (Science/ セクション)")
    print(f"💾 チェックポイント保存: {args.output_dir}")

    history = agent.train(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        callback=callbacks
    )
    
    # モデルの保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / 'ppo_consortium_model.zip'
    agent.save(str(model_path))
    
    # 訓練履歴の保存
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        serializable_history = convert_to_serializable(history)
        json.dump(serializable_history, f, indent=2)
    print(f"📊 訓練履歴保存: {history_path}")
    
    # 評価（単一環境で実行）
    print("📊 最終評価中...")
    eval_sim = dFBASimulator(**simulator_params)
    eval_env = ConsortiumEnv(simulator=eval_sim, **env_params)
    results = agent.evaluate(n_episodes=args.eval_episodes)
    
    # 結果の保存
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
        models = select_consortium_models(all_models)
    else:
        models = create_mock_models()
    
    # 初期パラメータ
    initial_biomass, initial_metabolites = get_initial_params(models)
    
    # シミュレーターのセットアップ
    sim = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=100.0,
        volume=1.0,
        dt=0.5
    )
    
    # RL環境の作成
    env = ConsortiumEnv(
        simulator=sim,
        max_time=args.max_steps * 0.1
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
    train_parser.add_argument('--resume-from', type=str, default=None,
                             help='再開するモデルのパス')
    train_parser.add_argument('--total-timesteps', type=int, default=100000,
                             help='総訓練ステップ数')
    train_parser.add_argument('--max-steps', type=int, default=840,
                             help='1エピソードの最大ステップ数 (dt=0.2なら840で168h)')
    train_parser.add_argument('--target-degradation', type=float, default=0.9,
                             help='目標ゴム分解率')
    train_parser.add_argument('--amino-acid-cost', type=float, default=0.1,
                             help='アミノ酸コスト係数')
    train_parser.add_argument('--nutrient-cost', type=float, default=0.01,
                             help='基本栄養素（グルコース等）コスト係数')
    train_parser.add_argument('--data-log-path', type=str, default=None,
                             help='FBA結果のログ保存パス (サロゲートモデル用)')
    train_parser.add_argument('--learning-rate', type=float, default=5e-5,
                             help='学習率')
    train_parser.add_argument('--n-steps', type=int, default=4096,
                             help='PPOのn_steps (各環境での収集ステップ数)')
    train_parser.add_argument('--batch-size', type=int, default=512,
                             help='PPOのbatch_size')
    train_parser.add_argument('--ent-coef', type=float, default=0.01,
                             help='PPOのentropy係数')
    train_parser.add_argument('--gamma', type=float, default=0.99,
                             help='割引率')
    train_parser.add_argument('--output-dir', type=str, default='outputs',
                             help='出力ディレクトリ')
    train_parser.add_argument('--log-interval', type=int, default=1,
                             help='ログ出力間隔 (PPO更新ごと)')
    train_parser.add_argument('--save-freq', type=int, default=5000,
                             help='チェックポイント保存頻度')
    train_parser.add_argument('--eval-episodes', type=int, default=5,
                             help='評価エピソード数')
    train_parser.add_argument('--eval-during-training', action='store_true',
                             help='訓練中に定期的に評価を実行')
    train_parser.add_argument('--eval-freq', type=int, default=5000,
                             help='訓練中評価の頻度')
    train_parser.add_argument('--n-envs', type=int, default=8,
                             help='並列環境数')
    train_parser.add_argument('--tensorboard-log', type=str, default='outputs/tensorboard',
                             help='TensorBoardのログ保存先')
    train_parser.add_argument('--linear-lr', action='store_true',
                             help='学習率の線形減衰を有効にする')
    
    # 評価モード
    eval_parser = subparsers.add_parser('evaluate', help='訓練済みエージェントを評価')
    eval_parser.add_argument('--model-path', type=str, required=True,
                            help='訓練済みモデルのパス')
    eval_parser.add_argument('--sbml-dir', type=str, help='SBMLファイルのディレクトリ')
    eval_parser.add_argument('--max-steps', type=int, default=200,
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
