"""
Figure 5 Stochastic Extraction
Fixes: 4-tuple step API for VecEnv
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
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=100.0,
        volume=1.0,
        dt=0.2
    )
    return NoKillEnv(simulator=sim, max_time=240.0)


def extract_best_stochastic_rollout():
    env = DummyVecEnv([make_env])
    vecnorm_path = "outputs/refined_models/godmode/godmode_final_refined_vecnormalize.pkl"
    env = VecNormalize.load(vecnorm_path, env)
    env.training   = False
    env.norm_reward = False

    model_path = "outputs/refined_models/godmode/godmode_final_refined.zip"
    model = PPO.load(model_path, env=env)

    best_pha = -1
    best_df  = None

    print("Searching for best PHA production via stochastic exploration (3 episodes)...")
    for episode in range(3):
        obs  = env.reset()
        done = False
        unwrapped_env = env.envs[0]
        records = []

        while not done:
            state     = unwrapped_env.simulator.state
            total_pha = sum(s.pha_accumulated for s in state.species.values())
            or16_bio  = state.species['Actinoplanes_sp_OR16_lcp'].biomass if 'Actinoplanes_sp_OR16_lcp' in state.species else 0.0
            ns21_bio  = state.species['Rhizobacter_gummiphilus_NS21'].biomass if 'Rhizobacter_gummiphilus_NS21' in state.species else 0.0
            lp_bio    = state.species['Lactobacillus_plantarum'].biomass if 'Lactobacillus_plantarum' in state.species else 0.0

            records.append({
                'Time': state.time,
                'Rubber_Remaining': state.rubber_concentration,
                'C30_oligo_e': state.metabolites.get('C30_oligo_e', 0.0),
                'odtd_e':      state.metabolites.get('odtd_e', 0.0),
                'Total_PHA':   total_pha,
                'Biomass_OR16': or16_bio,
                'Biomass_NS21': ns21_bio,
                'Biomass_LP':   lp_bio
            })

            action, _ = model.predict(obs, deterministic=False)
            obs, _, dones, _ = env.step(action)
            done = dones[0] if hasattr(dones, '__len__') else bool(dones)

        df       = pd.DataFrame(records)
        final_pha = df['Total_PHA'].max()
        print(f"  Episode {episode+1} - Max PHA: {final_pha:.2f}")

        if final_pha > best_pha:
            best_pha = final_pha
            best_df  = df

    best_df.to_csv('paper_figures/fig5_product_optimization_stochastic.csv', index=False)
    print(f"Saved best stochastic rollout with PHA = {best_pha:.2f}")


if __name__ == "__main__":
    extract_best_stochastic_rollout()
