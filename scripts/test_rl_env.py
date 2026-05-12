import numpy as np
from pathlib import Path
from main import load_sbml_models, select_consortium_models, get_initial_params
from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv

def test_rl_env():
    sbml_dir = Path('models/sbml/final_consortium')
    all_models = load_sbml_models(sbml_dir)
    models = select_consortium_models(all_models)
    initial_biomass, initial_metabolites = get_initial_params(models)
    
    sim = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=100.0,
        volume=1.0,
        dt=1.0, # 1 hour dt for testing
        carrying_capacity=20.0
    )
    
    env = ConsortiumEnv(simulator=sim)
    obs, info = env.reset()
    print("--- Environment Reset ---")
    print(f"Initial Obs shape: {obs.shape}")
    
    # Generate a random action in [-1.0, 1.0]
    action = np.random.uniform(-1.0, 1.0, size=(5,)).astype(np.float32)
    print(f"\n--- Applying Action ---")
    print(f"Action Raw ([-1, 1]): {action}")
    norm_action = (action + 1.0) / 2.0
    print(f"Action Scaled: NH4={norm_action[0]*10:.2f}, Arg={norm_action[1]*1:.2f}, Leu={norm_action[2]*1:.2f}, Trp={norm_action[3]*1:.2f}, kLa={norm_action[4]*200:.2f}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"\n--- Step Result ---")
    print(f"Reward: {reward:.4f}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    print("Info Dictionary:")
    for k, v in info.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    test_rl_env()