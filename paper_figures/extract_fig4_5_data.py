import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Ensure root dir is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rl_environment import ConsortiumEnv
from src.dfba_simulator import dFBASimulator
from main import load_sbml_models, select_consortium_models, get_initial_params

def extract_fig4_data():
    tb_path = "outputs/final_v15_do_monitoring/tensorboard/PPO_0"
    
    if not os.path.exists(tb_path):
        print(f"TensorBoard path {tb_path} not found.")
        return
        
    ea = EventAccumulator(tb_path)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    print(f"Available TensorBoard tags: {tags}")
    
    survival_tag = 'Science/Survival_Hours'
    rubber_tag = 'Science/Total_Rubber_Degraded'
    
    # Try to find exactly what was logged
    found_survival = None
    found_rubber = None
    for t in tags:
        if 'Survival' in t or 'survival' in t.lower():
            found_survival = t
        if 'Rubber' in t or 'degradation' in t.lower():
            found_rubber = t
            
    survival_tag = found_survival if found_survival else survival_tag
    rubber_tag = found_rubber if found_rubber else rubber_tag

    print(f"Using tags: survival={survival_tag}, rubber={rubber_tag}")

    df_s = pd.DataFrame({'step': [], 'survival_hours': []})
    df_r = pd.DataFrame({'step': [], 'total_rubber_degraded': []})

    if survival_tag in tags:
        try:
            survival_events = ea.Scalars(survival_tag)
            df_s = pd.DataFrame({
                'step': [e.step for e in survival_events],
                'survival_hours': [e.value for e in survival_events]
            })
        except Exception as e:
            print(f"Error loading {survival_tag}: {e}")

    if rubber_tag in tags:
        try:
            rubber_events = ea.Scalars(rubber_tag)
            df_r = pd.DataFrame({
                'step': [e.step for e in rubber_events],
                'total_rubber_degraded': [e.value for e in rubber_events]
            })
        except Exception as e:
            print(f"Error loading {rubber_tag}: {e}")

    if not df_s.empty and not df_r.empty:
        df = pd.merge_asof(df_s, df_r, on='step')
    elif not df_s.empty:
        df = df_s
        df['total_rubber_degraded'] = 0.0
    elif not df_r.empty:
        df = df_r
        df['survival_hours'] = 0.0
    else:
        df = pd.DataFrame(columns=['step', 'survival_hours', 'total_rubber_degraded'])

    if not df.empty:
        df['survival_hours_rolling'] = df['survival_hours'].rolling(window=10, min_periods=1).mean()
        df['total_rubber_degraded_rolling'] = df['total_rubber_degraded'].rolling(window=10, min_periods=1).mean()

    df.to_csv('paper_figures/fig4_survival_evolution.csv', index=False)
    print("Saved fig4_survival_evolution.csv")

def extract_fig5_data():
    model_path = "outputs/final_v15_do_monitoring/ppo_consortium_2000000_steps.zip"
    vecnorm_path = "outputs/final_v15_do_monitoring/ppo_consortium_vecnormalize_2000000_steps.pkl"
    
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found.")
        return
        
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
        return ConsortiumEnv(simulator=sim, max_time=168.0)
        
    env = DummyVecEnv([make_env])
    
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"Warning: vecnorm_path {vecnorm_path} not found. Running without normalization.")
        
    model = PPO.load(model_path, env=env)
    
    obs = env.reset()
    done = False
    
    # Access the unwrapped environment to get raw states
    unwrapped_env = env.envs[0]
    
    records = []
    
    while not done:
        state = unwrapped_env.simulator.state
        
        # Calculate PHA
        total_pha = sum(s.pha_accumulated for s in state.species.values())
        
        # Extract biomass robustly
        or16_biomass = state.species['Actinoplanes_sp_OR16_lcp'].biomass if 'Actinoplanes_sp_OR16_lcp' in state.species else 0.0
        ns21_biomass = state.species['Rhizobacter_gummiphilus_NS21'].biomass if 'Rhizobacter_gummiphilus_NS21' in state.species else 0.0
        lp_biomass = state.species['Lactobacillus_plantarum'].biomass if 'Lactobacillus_plantarum' in state.species else 0.0
        
        records.append({
            'Time': state.time,
            'Rubber_Remaining': state.rubber_concentration,
            'C30_oligo_e': state.metabolites.get('C30_oligo_e', 0.0),
            'odtd_e': state.metabolites.get('odtd_e', 0.0),
            'Total_PHA': total_pha,
            'Biomass_OR16': or16_biomass,
            'Biomass_NS21': ns21_biomass,
            'Biomass_LP': lp_biomass
        })
        
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        if isinstance(done, np.ndarray):
            done = done[0]
            
    df = pd.DataFrame(records)
    df.to_csv('paper_figures/fig5_product_optimization.csv', index=False)
    print("Saved fig5_product_optimization.csv")

if __name__ == "__main__":
    os.makedirs('paper_figures', exist_ok=True)
    print("Extracting Figure 4 data...")
    extract_fig4_data()
    print("Extracting Figure 5 data...")
    extract_fig5_data()
