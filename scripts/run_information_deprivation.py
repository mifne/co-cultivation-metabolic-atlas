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

def run_deprived_sim():
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
    
    env_orig = ConsortiumEnv(simulator=sim, max_time=168.0)
    dummy_vec = DummyVecEnv([lambda: env_orig])
    vec_norm = VecNormalize.load("outputs/checkpoints/ppo_godmode_v3_vecnormalize_550000_steps.pkl", dummy_vec)
    vec_norm.training = False
    
    model = PPO.load("outputs/checkpoints/ppo_godmode_v3_550000_steps.zip")
    
    obs, _ = env_orig.reset()
    history = []
    
    # 記録用関数
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
        # --- 情報剥奪マスクの作成 (Forced POMDP Constraints) ---
        # obs は本来16次元:
        # [0,1,2]: OR16, NS21, LP biomasses
        # [3]: pH, [4]: DO
        # [5..13]: metabolites (glc, rubber, C30, ODTD, PHA, BS, arg, trp, leu)
        # [14]: Time, [15]: Phase
        
        deprived_obs = obs.copy()
        
        # 1. 個別バイオマスの情報を消去し、総バイオマスを3等分して代入
        total_biomass = sum(s.biomass for s in sim.state.species.values())
        masked_biomass = (total_biomass / 3.0) / 20.0 # 20.0で正規化
        deprived_obs[0] = masked_biomass
        deprived_obs[1] = masked_biomass
        deprived_obs[2] = masked_biomass
        
        # 2. 細胞外代謝物（非観測変数）をデフォルト初期値（0.0）でマスク
        for idx in range(5, 14):
            deprived_obs[idx] = 0.0
            
        # pH, DO, Time, Phase はそのまま維持 (POMDPで観測可能な情報)
        
        # 観測値を正規化して予測
        norm_obs = vec_norm.normalize_obs(np.array([deprived_obs]))
        action, _ = model.predict(norm_obs, deterministic=True)
        
        # 生環境をステップ実行
        obs, reward, terminated, truncated, info = env_orig.step(action[0])
        
        history.append(record_state(sim.state))
        if terminated or truncated:
            break
            
    df = pd.DataFrame(history)
    df['Agent'] = 'God-Mode (Forced POMDP)'
    return df

def main():
    print("Running Information Deprivation Experiment on God-Mode Agent...")
    df_deprived = run_deprived_sim()
    
    # 既存のデータと結合
    df_existing = pd.read_csv('paper_figures/fig11_final_data.csv')
    # 重複を避けるために既存の 'God-Mode (Forced POMDP)' を削除
    df_existing = df_existing[df_existing['Agent'] != 'God-Mode (Forced POMDP)']
    
    df_final = pd.concat([df_existing, df_deprived])
    df_final.to_csv('paper_figures/fig11_final_data.csv', index=False)
    print("Experiment complete. Clean data saved to paper_figures/fig11_final_data.csv")

if __name__ == "__main__":
    main()
