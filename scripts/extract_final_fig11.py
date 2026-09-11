import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import load_sbml_models, select_consortium_models, get_initial_params
from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.rl_environment_pomdp import RealWorldConsortiumEnv

def run_final_sim(model_path, pkl_path, is_pomdp=False):
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
    
    # 評価用の生環境
    if is_pomdp:
        env_orig = RealWorldConsortiumEnv(simulator=sim, max_time=168.0)
    else:
        env_orig = ConsortiumEnv(simulator=sim, max_time=168.0)
    
    # 正規化統計だけをロードするためのダミー
    dummy_vec = DummyVecEnv([lambda: env_orig])
    vec_norm = VecNormalize.load(pkl_path, dummy_vec)
    vec_norm.training = False
    
    model = PPO.load(model_path)
    
    # Gymnasium形式でのループ（自動リセットを避ける）
    obs, _ = env_orig.reset()
    history = []
    
    # 初期状態の記録
    def record_state(s):
        return {
            'time': s.time,
            'rubber': s.rubber_concentration,
            'pha': sum(sp.pha_accumulated for sp in s.species.values()),
            'biomass_total': sum(sp.biomass for sp in s.species.values()),
            'biomass_OR16': s.species['Actinoplanes_sp_OR16_lcp'].biomass,
            'biomass_NS21': s.species['Rhizobacter_gummiphilus_NS21'].biomass,
            'biomass_LP': s.species['Lactobacillus_plantarum'].biomass,
            'pH': -np.log10(max(1e-12, s.metabolites.get('h_e', 0.0001)) / 1000.0)
        }
    
    history.append(record_state(sim.state))

    for _ in range(840):
        # 観測値を正規化して予測
        norm_obs = vec_norm.normalize_obs(np.array([obs]))
        action, _ = model.predict(norm_obs, deterministic=True)
        
        # 生環境をステップ実行
        obs, reward, terminated, truncated, info = env_orig.step(action[0])
        
        # 状態を記録
        history.append(record_state(sim.state))
        
        if terminated or truncated:
            break
        
    return pd.DataFrame(history)

def main():
    print("Re-extracting clean data (fixing overlap issue)...")
    
    # God-Mode
    df_gm = run_final_sim(
        "outputs/checkpoints/ppo_godmode_v3_550000_steps.zip",
        "outputs/checkpoints/ppo_godmode_v3_vecnormalize_550000_steps.pkl",
        is_pomdp=False
    )
    df_gm['Agent'] = 'God-Mode'
    
    # POMDP
    df_pomdp = run_final_sim(
        "outputs/refined_models/pomdp/pomdp_final_refined.zip",
        "outputs/refined_models/pomdp/pomdp_final_refined_vecnormalize.pkl",
        is_pomdp=True
    )
    df_pomdp['Agent'] = 'Real-World (POMDP)'
    
    df_final = pd.concat([df_gm, df_pomdp])
    output_path = "paper_figures/fig11_final_data.csv"
    df_final.to_csv(output_path, index=False)
    print(f"Clean data saved to {output_path}")

if __name__ == "__main__":
    main()
