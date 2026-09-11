import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
"""
Hyperparameter Optimization for dFBA-RL using Optuna
"""

import optuna
import numpy as np
from pathlib import Path
import argparse
import json
import torch

from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.ppo_agent import ConsortiumPPOAgent
from src.utils import load_sbml_models, select_consortium_models, get_initial_params

def objective(trial):
    # ハイパーパラメータのサンプリング
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    n_steps = trial.suggest_categorical("n_steps", [2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    ent_coef = trial.suggest_float("ent_coef", 0.01, 0.1, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.995)
    
    # 報酬重みのサンプリング（動的に環境へ反映）
    # ※ConsortiumEnvの__init__を拡張するか、外部から注入できるようにする必要があるが、
    # 今回は簡易的に試行ごとに環境変数をモックする手法を取るか、
    # あるいは ConsortiumEnv 側に重み引数を追加する
    
    # モデルとシミュレーターの準備
    sbml_dir = Path("models/sbml")
    all_models = load_sbml_models(sbml_dir)
    models = select_consortium_models(all_models)
    initial_biomass, initial_metabolites = get_initial_params(models)
    
    simulator_params = {
        'models': models,
        'initial_biomass': initial_biomass,
        'initial_metabolites': initial_metabolites,
        'initial_rubber': 100.0,
        'volume': 1.0,
        'dt': 0.5,
        'use_surrogate': True  # サロゲートモデルを使用
    }
    
    env_params = {
        'max_steps': 100,  # 評価時間を稼ぐために少し長めに
        'target_rubber_degradation': 0.8,
        'amino_acid_cost': 0.05,
        'nutrient_cost': 0.01
    }
    
    sim = dFBASimulator(**simulator_params)
    env = ConsortiumEnv(simulator=sim, **env_params)
    
    # エージェントの作成
    agent = ConsortiumPPOAgent(
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        gamma=gamma,
        n_envs=1,
        verbose=0
    )
    
    # 訓練 (サロゲートなので20,000ステップでも高速)
    try:
        agent.train(total_timesteps=20000)
        
        # 評価
        results = agent.evaluate(n_episodes=3)
        # 目標: ゴム分解率とLp維持のバランス。
        # 単純なmean_rewardを返すが、報酬関数自体にLp維持が組み込まれているため機能する。
        return results['mean_reward']
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return -5000.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=20)
    args = parser.parse_args()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)
    
    print("\n✨ Optimization Finished!")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")
    
    # 結果を保存
    output_dir = Path("outputs/optuna")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

if __name__ == "__main__":
    main()
