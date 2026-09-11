import sys, os, logging
logging.getLogger("cobra").setLevel(logging.ERROR)
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import load_sbml_models, select_consortium_models, get_initial_params
from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv

def run_eval():
    print("Starting evaluation...")
    sbml_dir = "models/sbml/final_consortium"
    all_models = load_sbml_models(Path(sbml_dir))
    models = select_consortium_models(all_models)
    initial_biomass, initial_metabolites = get_initial_params(models)

    def make_env():
        sim = dFBASimulator(
            models=models,
            initial_biomass=initial_biomass,
            initial_metabolites=initial_metabolites,
            initial_rubber=100.0,
            volume=1.0,
            dt=0.2
        )
        return ConsortiumEnv(simulator=sim, max_time=168.0)

    vec_path = "outputs/checkpoints/ppo_godmode_v3_vecnormalize_440000_steps.pkl"
    model_path = "outputs/checkpoints/ppo_godmode_v3_440000_steps.zip"

    if not os.path.exists(vec_path) or not os.path.exists(model_path):
        print("Missing files.")
        return
        
    dummy_env = DummyVecEnv([make_env])
    env = VecNormalize.load(vec_path, dummy_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)
    obs = env.reset()
    done = False
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        done = dones[0]
        step += 1
    
    info = infos[0]
    print("--- God-mode (New v3.0 Model - 440k steps) ---")
    print(f"Survival Hours: {info.get('survival_hours'):.2f} / 168.0 h")
    print(f"Total PHA Accumulated: {info.get('total_pha'):.2f} mmol")
    print(f"Rubber Remaining: {info.get('rubber_remaining'):.2f} g/L (Degraded: {info.get('total_rubber_degraded'):.2f} g/L)")
    print(f"Final pH: {info.get('ph'):.2f}")

if __name__ == "__main__":
    run_eval()
