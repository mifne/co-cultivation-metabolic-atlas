import os
import sys
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import load_sbml_models, select_consortium_models, get_initial_params, ConsortiumCallback
from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
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
        return ConsortiumEnv(simulator=sim, max_time=168.0)
    return _init

def main():
    os.makedirs('outputs/checkpoints', exist_ok=True)
    os.makedirs('outputs/tensorboard', exist_ok=True)

    num_cpu = 8
    print(f"Setting up {num_cpu} parallel environments for God-Mode training (Reward v3.0)...")
    
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    policy_kwargs = dict(net_arch=[256, 256])
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        tensorboard_log="outputs/tensorboard/",
        verbose=1,
        device=os.environ.get("PPO_DEVICE", "cpu")
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // num_cpu, 1),
        save_path='outputs/checkpoints/',
        name_prefix='ppo_godmode_v3',
        save_vecnormalize=True,  # VecNormalize統計をチェックポイントごとに保存
    )
    
    # --- 堅牢化修正: 最新のチェックポイントから再開 ---
    checkpoint_dir = Path("outputs/checkpoints")
    checkpoints = sorted(list(checkpoint_dir.glob("ppo_godmode_v3_*_steps.zip")), key=lambda x: int(x.stem.split('_')[-2])) if checkpoint_dir.exists() else []
    
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        print(f"🔄 Resuming God-Mode training from {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env, tensorboard_log="outputs/tensorboard/")
        stats_path = checkpoint_dir / latest_checkpoint.name.replace("ppo_godmode_v3", "ppo_godmode_v3_vecnormalize").replace(".zip", ".pkl")

        if stats_path.exists():
            env = VecNormalize.load(str(stats_path), env.venv)
            model.set_env(env)
    
    print("Starting Parallel Training of God-Mode Agent (Reward v3.1: PHA-Incentivized) on 8 cores...")
    model.learn(total_timesteps=300000, callback=[checkpoint_callback], reset_num_timesteps=False)


    model.save("outputs/checkpoints/ppo_godmode_v3_final")
    env.save("outputs/checkpoints/ppo_godmode_v3_vecnormalize_final.pkl")
    print("God-Mode Training Complete.")

if __name__ == "__main__":
    main()
