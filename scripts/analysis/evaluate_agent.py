import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.ppo_agent import ConsortiumPPOAgent
from src.utils import load_sbml_models, select_consortium_models, get_initial_params

def evaluate_and_plot(model_path: str, sbml_dir: str, output_dir: str):
    print(f"Loading models from {sbml_dir}...")
    all_models = load_sbml_models(Path(sbml_dir))
    models = select_consortium_models(all_models)
    
    initial_biomass, initial_metabolites = get_initial_params(models)
    
    sim = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=100.0,
        volume=1.0,
        dt=0.2
    )
    
    env = ConsortiumEnv(simulator=sim, max_time=672.0)
    
    print(f"Loading agent from {model_path}...")
    agent = ConsortiumPPOAgent(env=env)
    agent.load(model_path)
    
    # RLエージェントモード: VecNormalize環境を使用
    obs = agent.env.reset()
    
    history = {
        'time': [],
        'biomass_or16': [], 'biomass_ns21': [], 'biomass_lp': [],
        'rubber': [], 'pha': [],
        'action_or16': [], 'action_ns21': [], 'action_lp': [], 'action_yeast': [], 'action_kla': [],
        'reward': []
    }
    
    done = False
    print("Simulating deterministic episode (RL Agent)...")
    while not done:
        actual_env = agent.env.envs[0]
        current_time = actual_env.simulator.state.time
        history['time'].append(current_time)
        
        # 推論 (正規化されたobsを使用)
        action_batch, _ = agent.predict(obs, deterministic=True)
        action_val = action_batch[0]
        
        # --- 物理量への変換 (環境側のロジックと同期。再正規化は不要) ---
        if current_time < 48.0: max_feed_common = 0.5
        else: max_feed_common = 0.05
        max_feed_specific = 0.1
        
        history['action_or16'].append(float(action_val[0]) * max_feed_specific)
        history['action_ns21'].append(float(action_val[1]) * max_feed_specific)
        history['action_lp'].append(float(action_val[2]) * max_feed_specific)
        history['action_yeast'].append(float(action_val[3]) * max_feed_common)
        history['action_kla'].append(float(action_val[4]) * 200.0)
        
        # ステップ実行 (正規化環境)
        obs, rewards, dones, infos = agent.env.step(action_batch)
        info = infos[0]
        
        history['reward'].append(rewards[0])
        history['biomass_or16'].append(info.get('biomass_or16', 0.0))
        history['biomass_ns21'].append(info.get('biomass_ns21', 0.0))
        history['biomass_lp'].append(info.get('biomass_lp', 0.0))
        history['rubber'].append(info.get('rubber_remaining', 0.0))
        history['pha'].append(info.get('total_pha', 0.0))
        
        done = dones[0]

    print("Simulation complete. Generating plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    time_arr = np.array(history['time'])
    plt.style.use('bmh')
    
    # A. バイオマス
    plt.figure(figsize=(10, 6))
    plt.plot(time_arr, history['biomass_or16'], label='OR16 (Lcp)')
    plt.plot(time_arr, history['biomass_ns21'], label='NS21 (Rox + PHA)')
    plt.plot(time_arr, history['biomass_lp'], label='LP (Stabilizer)')
    plt.xlabel('Time (h)')
    plt.ylabel('Biomass (g/L)')
    plt.title('Consortium Biomass Dynamics')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'A_biomass.png'))
    plt.close()
    
    # B. ゴム & PHA
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(time_arr, history['rubber'], 'k-', label='Rubber')
    ax2.plot(time_arr, history['pha'], 'g-', label='PHA')
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Rubber Remaining (g/L)', color='k')
    ax2.set_ylabel('PHA Accumulated (mmol)', color='g')
    plt.title('Rubber Degradation and PHA Accumulation')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'C_rubber_pha.png'))
    plt.close()
    
    # D. アクション (ラベルを現在の仕様に統一)
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_arr, history['action_or16'], label='sn_or16')
    plt.plot(time_arr, history['action_ns21'], label='sn_ns21')
    plt.plot(time_arr, history['action_lp'], label='sn_lp')
    plt.plot(time_arr, history['action_yeast'], label='Yeast Extract', linestyle='--')
    plt.ylabel('Nutrient Addition (mM/step)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_arr, history['action_kla'], label='kLa (O2 Transfer)', color='cyan')
    plt.xlabel('Time (h)')
    plt.ylabel('kLa (1/h)')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle('Agent Actions Over Time')
    plt.savefig(os.path.join(output_dir, 'D_actions.png'))
    plt.close()
    
    print(f"All plots saved to {output_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--sbml-dir', type=str, default='models/sbml/final_consortium')
    parser.add_argument('--output-dir', type=str, default='results/rl_eval')
    args = parser.parse_args()
    
    evaluate_and_plot(args.model_path, args.sbml_dir, args.output_dir)
