import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rl_environment import ConsortiumEnv
from src.dfba_simulator import dFBASimulator
from main import load_sbml_models, select_consortium_models, get_initial_params

class NoKillEnv(ConsortiumEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # Override termination to ignore the 72h kill switch
        terminated = any(s.biomass < 0.01 for s in self.simulator.state.species.values())
        return obs, reward, terminated, truncated, info

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
    return NoKillEnv(simulator=sim, max_time=168.0)

env = DummyVecEnv([make_env])
vecnorm_path = "outputs/final_v15_do_monitoring/ppo_consortium_vecnormalize_2000000_steps.pkl"
env = VecNormalize.load(vecnorm_path, env)
env.training = False
env.norm_reward = False

model_path = "outputs/final_v15_do_monitoring/ppo_consortium_2000000_steps.zip"
model = PPO.load(model_path, env=env)

obs = env.reset()
done = False
unwrapped_env = env.envs[0]
records = []

while not done:
    state = unwrapped_env.simulator.state
    total_pha = sum(s.pha_accumulated for s in state.species.values())
    records.append({
        'Time': state.time,
        'Rubber_Remaining': state.rubber_concentration,
        'Total_PHA': total_pha,
        'Biomass_OR16': state.species['Actinoplanes_sp_OR16_lcp'].biomass,
        'pH': -np.log10(state.metabolites.get('h_e', 1e-4)/1000.0)
    })
    
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if isinstance(done, np.ndarray): done = done[0]

df = pd.DataFrame(records)
print(f"Max Rubber Degraded: {100.0 - df['Rubber_Remaining'].min():.2f}")
print(f"Max PHA: {df['Total_PHA'].max():.2f}")
print(f"Survival Time: {df['Time'].max():.2f}")
