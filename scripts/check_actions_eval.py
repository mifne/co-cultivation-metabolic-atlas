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

print("--- Deterministic Actions & pH ---")
for i in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if isinstance(done, np.ndarray): done = done[0]
    print(f"Step {i+1}: Action={np.round(action[0], 2)}, pH={info[0]['ph']:.2f}")

print("--- Stochastic Actions & pH ---")
obs = env.reset()
done = False
for i in range(10):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    if isinstance(done, np.ndarray): done = done[0]
    print(f"Step {i+1}: Action={np.round(action[0], 2)}, pH={info[0]['ph']:.2f}")

