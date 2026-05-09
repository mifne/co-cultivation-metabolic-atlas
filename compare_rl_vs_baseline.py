import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.ppo_agent import ConsortiumPPOAgent
from main import load_sbml_models, select_consortium_models, get_initial_params

def run_simulation(env, agent=None, static_action=None):
    """
    シミュレーションを実行。
    agentが指定されている場合は、agent.env（VecNormalize）を使用して正規化された推論を行う。
    """
    history = {
        'time': [],
        'rubber': [],
        'pha': [],
        'reward': [],
        'r_cost': []
    }

    if agent is not None:
        # RLエージェントモード: agent.env (VecNormalize) を使用して推論とステップを実行
        obs = agent.env.reset()
        done = False
        while not done:
            # 正規化されたobsを渡し、正規化された空間でのアクションを取得
            action_batch, _ = agent.predict(obs, deterministic=True)
            # ベクトル環境を1ステップ進める
            obs, rewards, dones, infos = agent.env.step(action_batch)
            
            info = infos[0]
            # 内部のシミュレーター状態にアクセスして時間を取得
            actual_env = agent.env.envs[0]
            
            history['time'].append(actual_env.simulator.state.time)
            history['reward'].append(rewards[0])
            history['rubber'].append(info.get('rubber_remaining', 0.0))
            history['pha'].append(info.get('total_pha', 0.0))
            history['r_cost'].append(info.get('r_cost', 0.0))
            
            done = dones[0]
    else:
        # ベースラインモード: 生のenvを使用
        obs, _ = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            history['time'].append(env.simulator.state.time)
            # アクション空間 [0, 1] の値を直接渡す
            obs, reward, done, truncated, info = env.step(static_action)
            
            history['reward'].append(reward)
            history['rubber'].append(info.get('rubber_remaining', 0.0))
            history['pha'].append(info.get('total_pha', 0.0))
            history['r_cost'].append(info.get('r_cost', 0.0))

    return history

def compare_and_plot(model_path: str, sbml_dir: str, output_dir: str):
    print(f"Loading models from {sbml_dir}...")
    all_models = load_sbml_models(Path(sbml_dir))
    models = select_consortium_models(all_models)
    
    initial_biomass, initial_metabolites = get_initial_params(models)
    
    # シミュレーター設定の共通化
    sim_params = {
        'models': models,
        'initial_biomass': initial_biomass,
        'initial_metabolites': initial_metabolites,
        'initial_rubber': 100.0,
        'volume': 1.0,
        'dt': 0.2
    }
    
    # 1. RLエージェントの実行
    print(f"Loading agent from {model_path}...")
    test_sim = dFBASimulator(**sim_params)
    test_env = ConsortiumEnv(simulator=test_sim, max_time=672.0)
    agent = ConsortiumPPOAgent(env=test_env)
    agent.load(model_path)
    
    print("Running RL Agent simulation (with VecNormalize)...")
    rl_history = run_simulation(None, agent=agent)
    
    # 2. ベースラインの実行
    print("Running Static Baseline simulation (25% constant feeding)...")
    base_sim = dFBASimulator(**sim_params)
    base_env = ConsortiumEnv(simulator=base_sim, max_time=672.0)
    # 修正: アクション空間 [0, 1] に合わせ、25%供給として 0.25 を設定
    static_action = np.array([0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    baseline_history = run_simulation(base_env, static_action=static_action)
    
    # プロット作成
    print("Generating comparison plots...")
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('bmh')
    
    # A. PHA蓄積量の比較
    plt.figure(figsize=(10, 6))
    plt.plot(rl_history['time'], rl_history['pha'], label=f"RL Agent (Final: {rl_history['pha'][-1]:.1f} mmol)", color='green', linewidth=2.5)
    plt.plot(baseline_history['time'], baseline_history['pha'], label=f"Static Baseline (Final: {baseline_history['pha'][-1]:.1f} mmol)", color='gray', linestyle='--')
    plt.xlabel('Time (h)', fontsize=12)
    plt.ylabel('PHA Accumulated (mmol)', fontsize=12)
    plt.title('Impact of Reinforcement Learning: PHA Production', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.fill_between(rl_history['time'], rl_history['pha'], baseline_history['pha'], color='green', alpha=0.1)
    plt.savefig(os.path.join(output_dir, 'RL_vs_Baseline_PHA.png'))
    plt.close()

    # B. ゴム分解の比較
    plt.figure(figsize=(10, 6))
    plt.plot(rl_history['time'], rl_history['rubber'], label=f"RL Agent (Remaining: {rl_history['rubber'][-1]:.1f} g/L)", color='red', linewidth=2.5)
    plt.plot(baseline_history['time'], baseline_history['rubber'], label=f"Static Baseline (Remaining: {baseline_history['rubber'][-1]:.1f} g/L)", color='gray', linestyle='--')
    plt.xlabel('Time (h)', fontsize=12)
    plt.ylabel('Rubber Remaining (g/L)', fontsize=12)
    plt.title('Impact of Reinforcement Learning: Rubber Degradation', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'RL_vs_Baseline_Rubber.png'))
    plt.close()

    # 数値データの書き出し
    df = pd.DataFrame({
        'Time (h)': rl_history['time'],
        'RL_Rubber_Remaining (g/L)': rl_history['rubber'],
        'Baseline_Rubber_Remaining (g/L)': baseline_history['rubber'],
        'RL_PHA_Accumulated (mmol)': rl_history['pha'],
        'Baseline_PHA_Accumulated (mmol)': baseline_history['pha']
    })
    csv_path = os.path.join(output_dir, 'numerical_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {output_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--sbml-dir', type=str, default='models/sbml/final_consortium')
    parser.add_argument('--output-dir', type=str, default='results/rl_comparison')
    args = parser.parse_args()
    
    compare_and_plot(args.model_path, args.sbml_dir, args.output_dir)
