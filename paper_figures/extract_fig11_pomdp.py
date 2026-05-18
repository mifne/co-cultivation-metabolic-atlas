import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import load_sbml_models, select_consortium_models, get_initial_params
from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.rl_environment_pomdp import RealWorldConsortiumEnv

def make_env_godmode():
    sbml_dir = "models/sbml/final_consortium"
    all_models = load_sbml_models(Path(sbml_dir))
    models = select_consortium_models(all_models)
    initial_biomass, initial_metabolites = get_initial_params(models)
    sim = dFBASimulator(models=models, initial_biomass=initial_biomass, initial_metabolites=initial_metabolites, initial_rubber=100.0, volume=1.0, dt=0.2)
    return ConsortiumEnv(simulator=sim, max_time=240.0)

def make_env_pomdp():
    sbml_dir = "models/sbml/final_consortium"
    all_models = load_sbml_models(Path(sbml_dir))
    models = select_consortium_models(all_models)
    initial_biomass, initial_metabolites = get_initial_params(models)
    sim = dFBASimulator(models=models, initial_biomass=initial_biomass, initial_metabolites=initial_metabolites, initial_rubber=100.0, volume=1.0, dt=0.2)
    return RealWorldConsortiumEnv(simulator=sim, max_time=240.0)

def extract_agent_data(agent_type, model_path, vecnorm_path):
    if agent_type == 'God-Mode':
        env = DummyVecEnv([make_env_godmode])
    else:
        env = DummyVecEnv([make_env_pomdp])
        
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return pd.DataFrame()

    obs = env.reset()
    done = False
    data = []
    
    base_env = env.venv.envs[0]
    
    while not done:
        sim_state = base_env.simulator.state
        time_t = sim_state.time
        
        action, _ = model.predict(obs, deterministic=True)
        current_action = action[0] if len(action.shape) > 1 else action
        
        ph = sim_state.metabolites.get('h_e', 1e-7)
        ph_val = -np.log10(max(1e-12, ph) / 1000.0)
        
        biomass_or16 = sim_state.species['Actinoplanes_sp_OR16_lcp'].biomass if 'Actinoplanes_sp_OR16_lcp' in sim_state.species else 0
        biomass_ns21 = sim_state.species['Rhizobacter_gummiphilus_NS21'].biomass if 'Rhizobacter_gummiphilus_NS21' in sim_state.species else 0
        biomass_lp = sim_state.species['Lactobacillus_plantarum'].biomass if 'Lactobacillus_plantarum' in sim_state.species else 0
        total_pha = sum(s.pha_accumulated for s in sim_state.species.values())
        
        data.append({
            'Time': time_t,
            'pH': ph_val,
            'Biomass_OR16': biomass_or16,
            'Biomass_NS21': biomass_ns21,
            'Biomass_LP': biomass_lp,
            'Total_PHA': total_pha,
            'Agent_Type': agent_type
        })
        
        obs, rewards, dones, infos = env.step(action)
        done = dones[0]
        
    return pd.DataFrame(data)

def main():
    os.makedirs('paper_figures', exist_ok=True)
    
    # 1. Extract God-Mode data
    print("Extracting God-Mode Agent Data...")
    df_god = extract_agent_data(
        'God-Mode', 
        "outputs/refined_models/godmode/godmode_final_refined.zip", 
        "outputs/refined_models/godmode/godmode_final_refined_vecnormalize.pkl"
    )
    
    # 2. Extract Real-World POMDP data
    print("Extracting Real-World POMDP Agent Data...")
    # Try refined model first, fall back to best raw checkpoint
    pomdp_zip = "outputs/refined_models/pomdp/pomdp_final_refined.zip"
    pomdp_pkl = "outputs/refined_models/pomdp/pomdp_final_refined_vecnormalize.pkl"
    if not os.path.exists(pomdp_zip):
        pomdp_zip = "outputs/checkpoints_pomdp/ppo_realworld_150000_steps.zip"
        pomdp_pkl = ""  # no vecnormalize for raw checkpoints
    df_pomdp = extract_agent_data(
        'Real-World (POMDP)',
        pomdp_zip,
        pomdp_pkl
    )
    
    df_combined = pd.concat([df_god, df_pomdp], ignore_index=True)
    df_combined.to_csv('paper_figures/fig11_pomdp_comparison.csv', index=False)
    print("Saved paper_figures/fig11_pomdp_comparison.csv")

if __name__ == "__main__":
    main()
