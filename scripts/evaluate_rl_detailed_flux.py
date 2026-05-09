"""
RL Agent Detailed Behavior & Flux Analysis
RLエージェントの詳細挙動と内部フラックスの解析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.ppo_agent import ConsortiumPPOAgent
from main import load_sbml_models, select_consortium_models, get_initial_params

def run_detailed_evaluation(model_path: str = 'outputs/ppo_consortium_model.zip', steps: int = 200):
    # 1. 環境とエージェントのセットアップ
    sbml_dir = Path('models/sbml/final_consortium')
    all_models = load_sbml_models(sbml_dir)
    models = select_consortium_models(all_models)
    initial_biomass, initial_metabolites = get_initial_params(models)
    
    sim = dFBASimulator(
        models=models, 
        initial_biomass=initial_biomass, 
        initial_metabolites=initial_metabolites, 
        initial_rubber=100.0,
        dt=1.0 # 解析用に1時間ステップにする
    )
    env = ConsortiumEnv(simulator=sim, max_time=steps)
    
    agent = ConsortiumPPOAgent(env=env)
    if Path(model_path).exists():
        print(f"📂 Loading trained agent from {model_path}")
        agent.load(model_path)
    else:
        print("⚠️  Trained agent not found. Using random actions for diagnostic purposes.")

    # 2. シミュレーション実行とデータ収集
    obs, _ = env.reset()
    history = []
    
    print(f"🚀 Running simulation for {steps} hours...")
    
    for t in range(steps):
        if Path(model_path).exists():
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 主要反応のフラックスを抽出
        # シミュレーターの内部状態にアクセス
        fluxes = {}
        for name, model in sim.models.items():
            # 最後に実行されたFBAの結果は保存されていないため、再度計算するか、
            # simulatorを改造して保存するようにする必要があるが、
            # ここでは簡易的に直近のSpeciesStateから情報を取る
            state = sim.state.species[name]
            fluxes[f'{name}_growth'] = state.growth_rate
            
            # 主要な分泌フラックス (metabolite_secretionから)
            fluxes[f'{name}_pha'] = state.metabolite_secretion.get('pha_c', 0.0)
            fluxes[f'{name}_phb'] = state.metabolite_secretion.get('phb_c', 0.0)
            fluxes[f'{name}_bs'] = state.metabolite_secretion.get('biosurfactant_e', 0.0)
            fluxes[f'{name}_rubber_uptake'] = state.metabolite_uptake.get('rubber_e', 0.0)

        record = {
            'time': sim.state.time,
            'reward': reward,
            'ph': info['ph'],
            'rubber': info['rubber_remaining'],
            **fluxes,
            'biomass_or16': info['biomass_or16'],
            'biomass_ns21': info['biomass_ns21'],
            'biomass_lp': info['biomass_lp']
        }
        history.append(record)
        
        if terminated or truncated:
            break

    df = pd.DataFrame(history)
    df.to_csv('scripts/rl_detailed_flux_analysis.csv', index=False)
    print("✅ Simulation complete. Results saved to scripts/rl_detailed_flux_analysis.csv")

    # 3. 可視化
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('RL Agent Behavior & Metabolic Flux Analysis', fontsize=16)

    # (0,0) Biomass & Rubber
    ax = axes[0, 0]
    ax.plot(df['time'], df['biomass_or16'], label='OR16')
    ax.plot(df['time'], df['biomass_ns21'], label='NS21')
    ax.plot(df['time'], df['biomass_lp'], label='LP')
    ax2 = ax.twinx()
    ax2.plot(df['time'], df['rubber'], 'k--', label='Rubber', alpha=0.5)
    ax.set_title('Biomass and Rubber Concentration')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Biomass (g/L)')
    ax2.set_ylabel('Rubber (g/L)')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # (0,1) pH
    ax = axes[0, 1]
    ax.plot(df['time'], df['ph'], color='green')
    ax.axhline(7.0, color='r', linestyle=':', alpha=0.3)
    ax.axhline(6.0, color='r', linestyle=':', alpha=0.3)
    ax.axhline(8.0, color='r', linestyle=':', alpha=0.3)
    ax.set_title('pH Stability')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('pH')
    ax.set_ylim(4, 10)

    # (1,0) Growth Rates
    ax = axes[1, 0]
    for col in [c for c in df.columns if '_growth' in c]:
        ax.plot(df['time'], df[col], label=col)
    ax.set_title('Growth Rates (mu)')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('1/h')
    ax.legend()

    # (1,1) PHA Production Flux
    ax = axes[1, 1]
    for col in [c for c in df.columns if '_pha' in c or '_phb' in c]:
        if df[col].sum() > 0:
            ax.plot(df['time'], df[col], label=col)
    ax.set_title('PHA/PHB Production Flux')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('mmol/gDW/h')
    ax.legend()

    # (2,0) BS Production Flux
    ax = axes[2, 0]
    for col in [c for c in df.columns if '_bs' in c]:
        if df[col].sum() > 0:
            ax.plot(df['time'], df[col], label=col)
    ax.set_title('Biosurfactant Production Flux')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('mmol/gDW/h')
    ax.legend()

    # (2,1) Reward
    ax = axes[2, 1]
    ax.plot(df['time'], df['reward'], color='purple')
    ax.set_title('Step Reward')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Reward')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('scripts/rl_detailed_flux_analysis.png')
    print("📊 Plot saved to scripts/rl_detailed_flux_analysis.png")

if __name__ == '__main__':
    run_detailed_evaluation()
