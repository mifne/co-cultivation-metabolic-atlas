import os
import sys
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import load_sbml_models, select_consortium_models, get_initial_params
from src.dfba_simulator import dFBASimulator
from src.rl_environment_pomdp import RealWorldConsortiumEnv
from src.gpu_assignment import assign_gpu_for_worker

def make_env(rank, seed=0):
    def _init():
        assign_gpu_for_worker(rank)
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
        env = RealWorldConsortiumEnv(simulator=sim, max_time=168.0)
        # Using the standard reset without seed argument as the wrapper might not handle it directly 
        # in some versions of gym, or it's handled internally. We just return the env.
        return env
    return _init

def main():
    os.makedirs('outputs/checkpoints_pomdp', exist_ok=True)
    os.makedirs('outputs/tensorboard_pomdp', exist_ok=True)

    num_cpu = 8  # Use 8 parallel CPU cores to dramatically speed up FBA solving
    print(f"Setting up {num_cpu} parallel environments for fast POMDP training...")
    
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    policy_kwargs = dict(net_arch=[128, 128])
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024, # Smaller steps per env to update more frequently with parallel envs
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        tensorboard_log="outputs/tensorboard_pomdp/",
        verbose=1,
        device="cpu"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // num_cpu, 1),
        save_path='outputs/checkpoints_pomdp/',
        name_prefix='ppo_realworld',
        save_vecnormalize=True,  # VecNormalize統計をチェックポイントごとに保存
    )
    
    # --- 堅牢化修正: 最新のチェックポイントから再開 ---
    latest_checkpoint = "outputs/checkpoints_pomdp/ppo_realworld_280000_steps.zip"
    if os.path.exists(latest_checkpoint):
        print(f"🔄 Resuming POMDP training from {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env, tensorboard_log="outputs/tensorboard_pomdp/")
        # VecNormalizeの統計情報も復元
        stats_path = "outputs/checkpoints_pomdp/ppo_realworld_vecnormalize_280000_steps.pkl"

        if os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, env.venv)
            model.set_env(env)

    print(f"Starting Parallel Training of Real-World POMDP Agent on {num_cpu} cores...")
    
    # Train for 1,000,000 steps
    model.learn(total_timesteps=1000000, callback=[checkpoint_callback], reset_num_timesteps=False)


    model.save("outputs/checkpoints_pomdp/ppo_realworld_final")
    env.save("outputs/checkpoints_pomdp/ppo_realworld_vecnormalize_final.pkl")
    print("Training Complete. Model saved to outputs/checkpoints_pomdp/")

if __name__ == "__main__":
    main()
