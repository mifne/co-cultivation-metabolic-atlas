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

def make_env_with_logging():
    sbml_dir = "models/sbml/final_consortium"
    all_models = load_sbml_models(Path(sbml_dir))
    models = select_consortium_models(all_models)
    initial_biomass, initial_metabolites = get_initial_params(models)
    
    # Enable internal dFBA flux logging!
    sim = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=100.0,
        volume=1.0,
        dt=0.2,
        data_log_path="paper_figures/fig9_raw_flux_log.csv"
    )
    return ConsortiumEnv(simulator=sim, max_time=240.0)

def main():
    model_path = "outputs/checkpoints/ppo_godmode_v3_550000_steps.zip"
    vecnorm_path = "outputs/checkpoints/ppo_godmode_v3_vecnormalize_550000_steps.pkl"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("Running episode to extract internal Metabolic Fluxes...")
    env = DummyVecEnv([make_env_with_logging])
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(model_path, env=env)
    obs = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        done = dones[0]

    print("Flux log generated at paper_figures/fig9_raw_flux_log.csv")

if __name__ == "__main__":
    main()
