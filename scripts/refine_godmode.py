import os
import sys
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import load_sbml_models, select_consortium_models, get_initial_params
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
    # 設定
    best_ckpt = "outputs/checkpoints/ppo_godmode_v3_150000_steps.zip"
    output_dir = "outputs/refined_models/godmode"
    log_dir = f"{output_dir}/logs"
    tb_log = f"{output_dir}/tensorboard"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    num_cpu = 8
    print(f"Resuming God-Mode training from {best_ckpt} for refinement...")
    
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    # 注意: 元の学習時にVecNormalize統計が保存されていなかったため、
    # ここでは新規に初期化します。学習が進むにつれて統計が再構築されます。
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // num_cpu,
        save_path=output_dir,
        name_prefix="godmode_refined",
        save_vecnormalize=True  # これが重要
    )

    model = PPO.load(best_ckpt, env=env, tensorboard_log=tb_log)
    
    # 10万ステップの追加学習（統計情報の安定化と保存が主目的）
    model.learn(
        total_timesteps=100000,
        callback=checkpoint_callback,
        reset_num_timesteps=False
    )
    
    # 最終モデルの保存
    model.save(f"{output_dir}/godmode_final_refined")
    env.save(f"{output_dir}/godmode_final_refined_vecnormalize.pkl")
    print(f"Refinement complete. Final model saved to {output_dir}")

if __name__ == "__main__":
    main()
