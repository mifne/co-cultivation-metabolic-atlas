import numpy as np
import pandas as pd
from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.ppo_agent import ConsortiumPPOAgent
from main import load_sbml_models, select_consortium_models, get_initial_params
from pathlib import Path

# 1. モデルと環境のロード
sbml_dir = Path('models/sbml/final_consortium')
all_models = load_sbml_models(sbml_dir)
models = select_consortium_models(all_models)
initial_biomass, initial_metabolites = get_initial_params(models)
sim = dFBASimulator(models=models, initial_biomass=initial_biomass, initial_metabolites=initial_metabolites, initial_rubber=100.0)
env = ConsortiumEnv(simulator=sim, max_time=100.0)

agent = ConsortiumPPOAgent(env=env)
agent.load('outputs/ppo_consortium_model.zip')

# 2. 実行とアクションの記録
obs, _ = env.reset()
history = []
for i in range(50):
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, term, trunc, info = env.step(action)
    history.append({
        'step': i,
        'action': action.tolist(),
        'reward': reward,
        'ph': info['ph'],
        'glc': sim.state.metabolites.get('glc__D_e', 0),
        'bio_or16': info['biomass_or16']
    })
    if term or trunc: break

df = pd.DataFrame(history)
print("--- Agent Behavior Analysis ---")
print(df[['step', 'ph', 'glc', 'bio_or16', 'reward']])
print("\nMean Action:", np.mean(df['action'].tolist(), axis=0))
