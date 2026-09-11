"""
Figure 5 Hybrid Strategy Extraction
Fixes: step API (4-tuple VecEnv), corrected termination condition
"""

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
    """全種死滅を終了条件とするが、単独死亡では終了しない環境"""
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        terminated = all(s.biomass < 0.01 for s in self.simulator.state.species.values())
        return obs, reward, terminated, truncated, info


def make_env():
    sbml_dir = "models/sbml/final_consortium"
    all_models = load_sbml_models(Path(sbml_dir))
    models     = select_consortium_models(all_models)
    initial_biomass, initial_metabolites = get_initial_params(models)
    sim = dFBASimulator(
        models=models, initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites, initial_rubber=100.0,
        volume=1.0, dt=0.2
    )
    return NoKillEnv(simulator=sim, max_time=672.0)


def extract_deterministic():
    """連続God-mode制御（決定論的）: ゴム分解→PHAの最大化"""
    env = DummyVecEnv([make_env])
    vecnorm_path = "outputs/checkpoints/ppo_godmode_v3_vecnormalize_550000_steps.pkl"
    env = VecNormalize.load(vecnorm_path, env)
    env.training   = False
    env.norm_reward = False
    model = PPO.load("outputs/checkpoints/ppo_godmode_v3_550000_steps.zip", env=env)

    obs  = env.reset()
    done = False
    records = []
    unwrapped_env = env.envs[0]

    while not done:
        state = unwrapped_env.simulator.state
        records.append({
            'Time': state.time,
            'Rubber_Remaining': state.rubber_concentration,
            'C30_oligo_e': state.metabolites.get('C30_oligo_e', 0.0),
            'odtd_e':      state.metabolites.get('odtd_e', 0.0),
            'Total_PHA': sum(s.pha_accumulated for s in state.species.values()),
            'YE_Feed': 0.0  # will update after action
        })
        action, _ = model.predict(obs, deterministic=True)
        # YE Feed = action[3] (common feed)
        a = action[0] if isinstance(action, np.ndarray) and len(action.shape) > 1 else action
        records[-1]['YE_Feed'] = float(a[3])
        obs, _, dones, _ = env.step(action)
        done = dones[0] if hasattr(dones, '__len__') else bool(dones)

    pd.DataFrame(records).to_csv('paper_figures/fig5_deterministic.csv', index=False)
    print("Saved deterministic run.")


def extract_hybrid():
    """ハイブリッド制御: 336h後に窒素飢餓を誘導してPHA最大化"""
    env = DummyVecEnv([make_env])
    vecnorm_path = "outputs/checkpoints/ppo_godmode_v3_vecnormalize_550000_steps.pkl"
    env = VecNormalize.load(vecnorm_path, env)
    env.training   = False
    env.norm_reward = False
    model = PPO.load("outputs/checkpoints/ppo_godmode_v3_550000_steps.zip", env=env)

    obs  = env.reset()
    done = False
    records = []
    unwrapped_env = env.envs[0]
    washed_out = False

    while not done:
        state = unwrapped_env.simulator.state
        records.append({
            'Time': state.time,
            'Rubber_Remaining': state.rubber_concentration,
            'C30_oligo_e': state.metabolites.get('C30_oligo_e', 0.0),
            'odtd_e':      state.metabolites.get('odtd_e', 0.0),
            'Total_PHA': sum(s.pha_accumulated for s in state.species.values()),
            'YE_Feed': 0.0
        })
        action, _ = model.predict(obs, deterministic=True)

        # ハイブリッドオーバーライド: 336h以降は窒素源をウォッシュアウト
        if state.time >= 336.0:
            if not washed_out:
                # 2段階発酵を模した窒素飢餓
                for n_met in ['nh4_e', 'yeast_extract_e', 'arg__L_e', 'trp__L_e', 'leu__L_e']:
                    unwrapped_env.simulator.state.metabolites[n_met] = 0.0
                washed_out = True

            # 窒素供給アクションを強制的にゼロに
            if isinstance(action, np.ndarray) and len(action.shape) == 2:
                action[0][0] = 0.0  # OR16 spec
                action[0][1] = 0.0  # NS21 spec
                action[0][2] = 0.0  # LP spec
                action[0][3] = 0.0  # YE (common)
                a = action[0]
            else:
                action[0] = 0.0; action[1] = 0.0; action[2] = 0.0; action[3] = 0.0
                a = action
        else:
            a = action[0] if isinstance(action, np.ndarray) and len(action.shape) == 2 else action

        records[-1]['YE_Feed'] = float(a[3])
        obs, _, dones, _ = env.step(action)
        done = dones[0] if hasattr(dones, '__len__') else bool(dones)

    pd.DataFrame(records).to_csv('paper_figures/fig5_hybrid.csv', index=False)
    print("Saved hybrid run.")


if __name__ == '__main__':
    extract_deterministic()
    extract_hybrid()
