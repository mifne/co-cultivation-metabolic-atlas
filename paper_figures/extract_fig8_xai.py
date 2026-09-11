import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import load_sbml_models, select_consortium_models, get_initial_params
from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv

def make_env():
    sbml_dir = "models/sbml/final_consortium"
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
    return ConsortiumEnv(simulator=sim, max_time=240.0)

def main():
    model_path = "outputs/checkpoints/ppo_godmode_v3_550000_steps.zip"
    vecnorm_path = "outputs/checkpoints/ppo_godmode_v3_vecnormalize_550000_steps.pkl"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    env = DummyVecEnv([make_env])
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(model_path, env=env)
    
    print("Running episode to collect state-action pairs for XAI...")
    obs = env.reset()
    done = False
    
    observations = []
    actions = []
    raw_states = []

    unwrapped_env = env.envs[0]

    while not done:
        # Save exact observation seen by agent
        observations.append(obs[0].copy())
        
        # Save raw meaningful states for pacing analysis (Task 3)
        state = unwrapped_env.simulator.state
        raw_states.append({
            'Time': state.time,
            'C30_oligo_e': state.metabolites.get('C30_oligo_e', 0.0),
            'Biomass_OR16': state.species['Actinoplanes_sp_OR16_lcp'].biomass if 'Actinoplanes_sp_OR16_lcp' in state.species else 0.0,
        })
        
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action[0].copy())
        
        obs, rewards, dones, infos = env.step(action)
        done = dones[0]

    X = np.array(observations)
    Y = np.array(actions)
    
    feature_names = [
        "Biomass OR16", "Biomass NS21", "Biomass LP", "pH", "Dissolved O2", 
        "Glucose", "Rubber", "C30 Oligomer", "ODTD", "PHA", 
        "Biosurfactant", "Arginine", "Tryptophan", "Leucine", "Time", "Phase"
    ]
    
    action_names = [
        "Feed OR16 (Action 0)", "Feed NS21 (Action 1)", "Feed LP (Action 2)", 
        "Common Feed (Action 3)", "Aeration (Action 4)"
    ]

    print("Training Surrogate Random Forest Models to extract Feature Importance...")
    importance_data = []

    for i in range(5): # 5 actions
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, Y[:, i])
        importances = rf.feature_importances_
        
        for j, imp in enumerate(importances):
            importance_data.append({
                'Action': action_names[i],
                'Feature': feature_names[j],
                'Importance': imp
            })

    df_importance = pd.DataFrame(importance_data)
    os.makedirs('paper_figures', exist_ok=True)
    df_importance.to_csv('paper_figures/fig8_feature_importance.csv', index=False)
    print("Saved paper_figures/fig8_feature_importance.csv")

    # Save raw states + actions for Pacing analysis (Figure 10)
    df_pacing = pd.DataFrame(raw_states)
    df_pacing['Feed_OR16'] = Y[:, 0]
    df_pacing.to_csv('paper_figures/fig10_pacing_data.csv', index=False)
    print("Saved paper_figures/fig10_pacing_data.csv")

if __name__ == "__main__":
    main()
