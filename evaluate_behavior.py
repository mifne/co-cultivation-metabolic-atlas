import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path

from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.ppo_agent import ConsortiumPPOAgent
from main import load_sbml_models, select_consortium_models, get_initial_params

def evaluate_behavior(model_path, sbml_dir, output_dir, run_baseline=False):
    # 1. モデルと環境のロード
    sbml_dir_path = Path(sbml_dir)
    all_models = load_sbml_models(sbml_dir_path)
    models = select_consortium_models(all_models)
    initial_biomass, initial_metabolites = get_initial_params(models)
    
    sim = dFBASimulator(
        models=models, 
        initial_biomass=initial_biomass, 
        initial_metabolites=initial_metabolites, 
        initial_rubber=100.0, 
        dt=0.2
    )
    env = ConsortiumEnv(simulator=sim, max_time=672.0)

    agent = ConsortiumPPOAgent(env=env)
    agent.load(model_path)

    history = []
    
    if run_baseline:
        # ベースラインモード
        obs, _ = env.reset()
        done = False
        truncated = False
        static_action = np.array([0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        
        while not (done or truncated):
            current_time = sim.state.time
            action_val = static_action
            obs, reward, done, truncated, info = env.step(action_val)
            
            # 物理量への変換 (環境側の設定と同期)
            if current_time < 48.0: max_feed_common = 0.5
            else: max_feed_common = 0.05
            max_feed_specific = 0.1
            
            history.append({
                'time': current_time,
                'sn_or16': float(action_val[0]) * max_feed_specific,
                'sn_ns21': float(action_val[1]) * max_feed_specific,
                'sn_lp': float(action_val[2]) * max_feed_specific,
                'yeast_extract': float(action_val[3]) * max_feed_common,
                'dynamic_kla': float(action_val[4]) * 200.0,
                'reward': reward,
                'total_pha': info.get('total_pha', 0),
                'rubber_remaining': info.get('rubber_remaining', 100.0),
                'ph': info.get('ph', 7.0)
            })
    else:
        # RLエージェントモード: VecNormalizeを使用
        obs = agent.env.reset()
        done = False
        while not done:
            actual_env = agent.env.envs[0]
            current_time = actual_env.simulator.state.time
            
            # 推論 (正規化されたobsを使用)
            action_val_batch, _ = agent.predict(obs, deterministic=True)
            action_val = action_val_batch[0]
            
            # ステップ実行 (正規化環境)
            obs, rewards, dones, infos = agent.env.step(action_val_batch)
            info = infos[0]
            
            # 物理量への変換 (修正: 再正規化を削除し、action_valを直接使用)
            if current_time < 48.0: max_feed_common = 0.5
            else: max_feed_common = 0.05
            max_feed_specific = 0.1
            
            history.append({
                'time': current_time,
                'sn_or16': float(action_val[0]) * max_feed_specific,
                'sn_ns21': float(action_val[1]) * max_feed_specific,
                'sn_lp': float(action_val[2]) * max_feed_specific,
                'yeast_extract': float(action_val[3]) * max_feed_common,
                'dynamic_kla': float(action_val[4]) * 200.0,
                'reward': rewards[0],
                'total_pha': info.get('total_pha', 0),
                'rubber_remaining': info.get('rubber_remaining', 100.0),
                'ph': info.get('ph', 7.0)
            })
            done = dones[0]

    df = pd.DataFrame(history)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'agent_behavior.csv'), index=False)
    
    print(f"--- Behavior Analysis ({'Baseline' if run_baseline else 'RL Agent'}) ---")
    print(f"Final PHA: {df['total_pha'].iloc[-1]:.2f} mmol")
    print(f"Final Rubber: {df['rubber_remaining'].iloc[-1]:.2f} g/L")
    
    # グラフ作成
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['sn_or16'], label='sn_or16')
    plt.plot(df['time'], df['sn_ns21'], label='sn_ns21')
    plt.plot(df['time'], df['sn_lp'], label='sn_lp')
    plt.plot(df['time'], df['yeast_extract'], label='yeast_extract')
    plt.xlabel('Time (h)')
    plt.ylabel('Feed Rate (mM/step)')
    plt.title('Agent Feeding Behavior')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'D_actions.png'))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--sbml-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--baseline', action='store_true', help='Run baseline instead of agent')
    args = parser.parse_args()
    
    evaluate_behavior(args.model_path, args.sbml_dir, args.output_dir, run_baseline=args.baseline)
