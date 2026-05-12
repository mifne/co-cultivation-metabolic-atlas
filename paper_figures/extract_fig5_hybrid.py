import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rl_environment import ConsortiumEnv
from src.dfba_simulator import dFBASimulator
from main import load_sbml_models, select_consortium_models, get_initial_params

class NoKillEnv(ConsortiumEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        terminated = any(s.biomass < 0.01 for s in self.simulator.state.species.values())
        return obs, reward, terminated, truncated, info

def make_env():
    sbml_dir = "models/sbml/final_consortium"
    all_models = load_sbml_models(Path(sbml_dir))
    models = select_consortium_models(all_models)
    initial_biomass, initial_metabolites = get_initial_params(models)
    sim = dFBASimulator(
        models=models, initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites, initial_rubber=100.0,
        volume=1.0, dt=0.2
    )
    return NoKillEnv(simulator=sim, max_time=168.0)

def extract_deterministic():
    env = DummyVecEnv([make_env])
    vecnorm_path = "outputs/final_v15_do_monitoring/ppo_consortium_vecnormalize_2000000_steps.pkl"
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False
    model = PPO.load("outputs/final_v15_do_monitoring/ppo_consortium_2000000_steps.zip", env=env)

    obs = env.reset()
    done = False
    records = []
    unwrapped_env = env.envs[0]

    while not done:
        state = unwrapped_env.simulator.state
        records.append({
            'Time': state.time, 'Rubber_Remaining': state.rubber_concentration,
            'C30_oligo_e': state.metabolites.get('C30_oligo_e', 0.0),
            'odtd_e': state.metabolites.get('odtd_e', 0.0),
            'Total_PHA': sum(s.pha_accumulated for s in state.species.values()),
            'YE_Feed': 0.0 # will update after action
        })
        action, _ = model.predict(obs, deterministic=True)
        records[-1]['YE_Feed'] = action[0][3] if isinstance(action, tuple) else action[0][3]
        obs, _, done, _ = env.step(action)
        if isinstance(done, np.ndarray): done = done[0]
        
    pd.DataFrame(records).to_csv('paper_figures/fig5_deterministic.csv', index=False)
    print("Saved deterministic run.")

def extract_hybrid():
    env = DummyVecEnv([make_env])
    vecnorm_path = "outputs/final_v15_do_monitoring/ppo_consortium_vecnormalize_2000000_steps.pkl"
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False
    model = PPO.load("outputs/final_v15_do_monitoring/ppo_consortium_2000000_steps.zip", env=env)

    obs = env.reset()
    done = False
    records = []
    unwrapped_env = env.envs[0]
    washed_out = False

    while not done:
        state = unwrapped_env.simulator.state
        records.append({
            'Time': state.time, 'Rubber_Remaining': state.rubber_concentration,
            'C30_oligo_e': state.metabolites.get('C30_oligo_e', 0.0),
            'odtd_e': state.metabolites.get('odtd_e', 0.0),
            'Total_PHA': sum(s.pha_accumulated for s in state.species.values()),
            'YE_Feed': 0.0
        })
        action, _ = model.predict(obs, deterministic=True)
        
        # HYBRID OVERRIDE: Force starvation after 80 hours
        if state.time >= 80.0:
            # Wash out nitrogen sources (mimicking 2-stage fermentation)
            if not washed_out:
                unwrapped_env.simulator.state.metabolites['nh4_e'] = 0.0
                unwrapped_env.simulator.state.metabolites['yeast_extract_e'] = 0.0
                for aa in ['arg__L_e', 'trp__L_e', 'leu__L_e', 'ile__L_e', 'val__L_e', 'lys__L_e', 'met__L_e', 'phe__L_e', 'his__L_e', 'tyr__L_e', 'thr__L_e', 'cys__L_e', 'ala__L_e', 'asp__L_e', 'glu__L_e', 'gly_e', 'pro__L_e', 'ser__L_e', 'asn__L_e', 'gln__L_e']:
                    unwrapped_env.simulator.state.metabolites[aa] = 0.0
                washed_out = True
                    
            if isinstance(action, np.ndarray) and len(action.shape) == 2:
                action[0][3] = 0.0 # Cut YE
                action[0][0] = 0.0 # Cut OR16 spec
                action[0][1] = 0.0 # Cut NS21 spec
                action[0][2] = 0.0 # Cut LP spec
            else:
                action[3] = 0.0
                action[0] = 0.0
                action[1] = 0.0
                action[2] = 0.0
                
        records[-1]['YE_Feed'] = action[0][3] if isinstance(action, np.ndarray) and len(action.shape) == 2 else action[3]
        obs, _, done, _ = env.step(action)
        if isinstance(done, np.ndarray): done = done[0]
        
    pd.DataFrame(records).to_csv('paper_figures/fig5_hybrid.csv', index=False)
    print("Saved hybrid run.")

if __name__ == '__main__':
    extract_deterministic()
    extract_hybrid()
