import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from main import make_env

def extract_2a():
    print("Extracting 2A...")
    log_dir = "outputs/final_v15_do_monitoring/tensorboard/PPO_0"
    if not os.path.exists(log_dir):
        print(f"Directory {log_dir} does not exist.")
        return
    ea = EventAccumulator(log_dir)
    ea.Reload()

    surv = ea.Scalars('Science/Survival_Hours') if 'Science/Survival_Hours' in ea.Tags().get('scalars', []) else []
    deg = ea.Scalars('Science/Total_Rubber_Degraded') if 'Science/Total_Rubber_Degraded' in ea.Tags().get('scalars', []) else []

    surv_df = pd.DataFrame([(s.step, s.value) for s in surv], columns=['Step', 'Survival_Hours'])
    deg_df = pd.DataFrame([(s.step, s.value) for s in deg], columns=['Step', 'Total_Rubber_Degraded'])

    if not surv_df.empty and not deg_df.empty:
        df = pd.merge(surv_df, deg_df, on='Step', how='outer').sort_values('Step')
    elif not surv_df.empty:
        df = surv_df
    elif not deg_df.empty:
        df = deg_df
    else:
        df = pd.DataFrame()

    os.makedirs('paper_figures', exist_ok=True)
    df.to_csv('paper_figures/fig2A_learning_curve.csv', index=False)
    print("Saved 2A.")

def extract_2b():
    print("Extracting 2B...")
    model_path = "outputs/final_v15_do_monitoring/ppo_consortium_2000000_steps.zip"
    vec_path = "outputs/final_v15_do_monitoring/ppo_consortium_vecnormalize_2000000_steps.pkl"
    sbml_dir = "models/sbml/final_consortium"

    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found.")
        return

    env_params = {'max_time': 168.0}
    env_fn = make_env(sbml_dir, env_params)
    dummy_env = DummyVecEnv([env_fn])
    env = VecNormalize.load(vec_path, dummy_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)
    base_env = env.venv.envs[0]
    
    obs = env.reset()
    
    data = []
    done = False
    
    while not done:
        sim_state = base_env.simulator.state
        time_t = sim_state.time
        biomass_or16 = sim_state.species['Actinoplanes_sp_OR16_lcp'].biomass if 'Actinoplanes_sp_OR16_lcp' in sim_state.species else 0
        biomass_ns21 = sim_state.species['Rhizobacter_gummiphilus_NS21'].biomass if 'Rhizobacter_gummiphilus_NS21' in sim_state.species else 0
        biomass_lp = sim_state.species['Lactobacillus_plantarum'].biomass if 'Lactobacillus_plantarum' in sim_state.species else 0
        ph = sim_state.metabolites.get('h_e', 1e-7)
        ph_val = -np.log10(max(1e-12, ph) / 1000.0)
        do_val = sim_state.metabolites.get('o2_e', 0)
        rubber = getattr(sim_state, 'rubber_concentration', 0)
        pha = sum(s.pha_accumulated for s in sim_state.species.values())

        action, _states = model.predict(obs, deterministic=True)
        
        row = {
            'Time': time_t,
            'Biomass_OR16': biomass_or16,
            'Biomass_NS21': biomass_ns21,
            'Biomass_LP': biomass_lp,
            'pH': ph_val,
            'DO': do_val,
            'Rubber_Remaining': rubber,
            'Total_PHA': pha,
            'Action_0': action[0][0] if len(action.shape) > 1 else action[0],
            'Action_1': action[0][1] if len(action.shape) > 1 else action[1],
            'Action_2': action[0][2] if len(action.shape) > 1 else action[2],
            'Action_3': action[0][3] if len(action.shape) > 1 else action[3],
            'Action_4': action[0][4] if len(action.shape) > 1 else action[4],
        }
        data.append(row)

        obs, rewards, dones, infos = env.step(action)
        done = dones[0]

    df = pd.DataFrame(data)
    df.to_csv('paper_figures/fig2B_timecourse.csv', index=False)
    print("Saved 2B.")

def extract_2cd():
    print("Extracting 2C/D...")
    model_path = "outputs/final_v15_do_monitoring/ppo_consortium_2000000_steps.zip"
    vec_path = "outputs/final_v15_do_monitoring/ppo_consortium_vecnormalize_2000000_steps.pkl"
    sbml_dir = "models/sbml/final_consortium"

    if not os.path.exists(model_path):
        return

    env_params = {'max_time': 168.0}
    env_fn = make_env(sbml_dir, env_params)
    dummy_env = DummyVecEnv([env_fn])
    env = VecNormalize.load(vec_path, dummy_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)
    base_env = env.venv.envs[0]
    
    env.reset()
    
    or16_vals = np.linspace(0.0, 5.0, 50)
    ns21_vals = np.linspace(0.0, 5.0, 50)
    
    results = []
    
    for or16 in or16_vals:
        for ns21 in ns21_vals:
            sim_state = base_env.simulator.state
            if 'Actinoplanes_sp_OR16_lcp' in sim_state.species:
                sim_state.species['Actinoplanes_sp_OR16_lcp'].biomass = or16
            if 'Rhizobacter_gummiphilus_NS21' in sim_state.species:
                sim_state.species['Rhizobacter_gummiphilus_NS21'].biomass = ns21
            if 'Lactobacillus_plantarum' in sim_state.species:
                sim_state.species['Lactobacillus_plantarum'].biomass = 0.1
                
            sim_state.metabolites['h_e'] = 10**(-7.0) * 1000.0
            sim_state.metabolites['o2_e'] = 0.25
            sim_state.metabolites['glc__D_e'] = 0.0
            sim_state.metabolites['C30_oligo_e'] = 5.0
            sim_state.metabolites['odtd_e'] = 5.0
            sim_state.metabolites['2mba_e'] = 0.01
            sim_state.metabolites['biosurfactant_e'] = 0.01
            
            sim_state.rubber_concentration = 50.0
            sim_state.time = 84.0
            for s in sim_state.species.values():
                s.pha_accumulated = 10.0 / len(sim_state.species) if len(sim_state.species) > 0 else 0
            
            unscaled_obs = base_env._get_observation(sim_state)
            unscaled_obs_batch = np.array([unscaled_obs], dtype=np.float32)
            
            norm_obs = env.normalize_obs(unscaled_obs_batch)
            
            action, _ = model.predict(norm_obs, deterministic=True)
            obs_tensor = torch.tensor(norm_obs, dtype=torch.float32).to(model.device)
            value = model.policy.predict_values(obs_tensor).item()
            
            results.append({
                'OR16_biomass': or16,
                'NS21_biomass': ns21,
                'Value': value,
                'Action_0': action[0][0] if len(action.shape) > 1 else action[0],
                'Action_1': action[0][1] if len(action.shape) > 1 else action[1],
                'Action_2': action[0][2] if len(action.shape) > 1 else action[2],
                'Action_3': action[0][3] if len(action.shape) > 1 else action[3],
                'Action_4': action[0][4] if len(action.shape) > 1 else action[4],
            })
            
    df = pd.DataFrame(results)
    df.to_csv('paper_figures/fig2CD_heatmap.csv', index=False)
    print("Saved 2CD.")

if __name__ == "__main__":
    extract_2a()
    extract_2b()
    extract_2cd()
