import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.ppo_agent import ConsortiumPPOAgent
from src.utils import load_sbml_models, select_consortium_models, get_initial_params

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

    def record(info, reward):
        # Both paths record the same physical endpoint. VecEnv has already
        # reset its live simulator when done=True, so use transition info.
        history['time'].append(float(info['time_h']))
        history['reward'].append(float(reward))
        history['rubber'].append(info.get('rubber_remaining', 0.0))
        history['pha'].append(info.get('total_pha', 0.0))
        history['r_cost'].append(info.get('r_cost', 0.0))

    if agent is not None:
        # RLエージェントモード: agent.env (VecNormalize) を使用して推論とステップを実行
        old_training, old_norm_reward = agent.env.training, agent.env.norm_reward
        agent.env.training, agent.env.norm_reward = False, False
        try:
            obs = agent.env.reset()
            done = False
            while not done:
                action_batch, _ = agent.predict(obs, deterministic=True)
                obs, rewards, dones, infos = agent.env.step(action_batch)
                record(infos[0], rewards[0])
                done = dones[0]
        finally:
            agent.env.training, agent.env.norm_reward = old_training, old_norm_reward
    else:
        # ベースラインモード: 生のenvを使用
        obs, _ = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            # アクション空間 [0, 1] の値を直接渡す
            obs, reward, done, truncated, info = env.step(static_action)
            record(info, reward)

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
    # Preserve different termination times as missing data instead of pairing
    # rows by position or extending a terminated culture's trajectory.
    df = pd.DataFrame({
        'Time (h)': rl_history['time'],
        'RL_Rubber_Remaining (g/L)': rl_history['rubber'],
        'RL_PHA_Accumulated (mmol/L)': rl_history['pha'],
    }).merge(pd.DataFrame({
        'Time (h)': baseline_history['time'],
        'Baseline_Rubber_Remaining (g/L)': baseline_history['rubber'],
        'Baseline_PHA_Accumulated (mmol/L)': baseline_history['pha'],
    }), on='Time (h)', how='outer').sort_values('Time (h)')
    
    # プロット作成
    print("Generating comparison plots...")
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('bmh')
    
    # A. PHA蓄積量の比較
    plt.figure(figsize=(10, 6))
    plt.plot(rl_history['time'], rl_history['pha'], label=f"RL Agent (Final: {rl_history['pha'][-1]:.1f} mmol/L)", color='green', linewidth=2.5)
    plt.plot(baseline_history['time'], baseline_history['pha'], label=f"Static Baseline (Final: {baseline_history['pha'][-1]:.1f} mmol/L)", color='gray', linestyle='--')
    plt.xlabel('Time (h)', fontsize=12)
    plt.ylabel('PHA Accumulated (mmol/L)', fontsize=12)
    plt.title('Impact of Reinforcement Learning: PHA Production', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    overlap = df.dropna(subset=['RL_PHA_Accumulated (mmol/L)', 'Baseline_PHA_Accumulated (mmol/L)'])
    plt.fill_between(overlap['Time (h)'], overlap['RL_PHA_Accumulated (mmol/L)'],
                     overlap['Baseline_PHA_Accumulated (mmol/L)'], color='green', alpha=0.1)
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
