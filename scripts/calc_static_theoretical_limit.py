"""
Calculate Static Theoretical Limit of Rubber Degradation
OR16モデルの静的な最大ゴム分解ポテンシャルをFBAを用いて計算する
"""

import cobra
import numpy as np
from pathlib import Path
from scipy.integrate import odeint

def calculate_static_limits():
    model_path = Path('models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml')
    model = cobra.io.read_sbml_model(str(model_path))
    
    print("--- Static FBA Limit Analysis ---")
    
    # 1. 境界条件を最大（無尽蔵）に開く
    with model:
        # 一旦すべての取り込みを閉じる
        for r in model.exchanges:
            if r.lower_bound < 0:
                r.lower_bound = 0
                
        # 必須無機塩類・酸素・水のみ無尽蔵に開く
        minimal_media = ['EX_h2o_e', 'EX_h_e', 'EX_nh4_e', 'EX_o2_e', 'EX_pi_e', 'EX_so4_e', 
                         'EX_ca2_e', 'EX_mg2_e', 'EX_fe3_e', 'EX_mn2_e', 'EX_zn2_e', 'EX_cu2_e', 
                         'EX_cobalt2_e', 'EX_cl_e', 'EX_k_e', 'EX_na_e', 'EX_ni2_e']
        for met in minimal_media:
            try:
                model.reactions.get_by_id(met).lower_bound = -1000
            except KeyError:
                pass
                
        # ゴム取り込みも無尽蔵にする
        model.reactions.get_by_id('EX_rubber_bulk_e').lower_bound = -1000
        # R_LCP の上限も開く
        model.reactions.get_by_id('R_LCP').upper_bound = 1000
        
        # A. ゴム分解フラックスの絶対最大値 (Growth 無視)
        model.objective = 'R_LCP'
        sol_max_lcp = model.optimize()
        max_lcp_flux = sol_max_lcp.objective_value
        max_rubber_uptake_abs = abs(sol_max_lcp.fluxes['EX_rubber_bulk_e'])
        print(f"Max Absolute LCP Flux: {max_lcp_flux:.4f} mmol/gDW/h")
        print(f"Max Rubber Uptake Flux: {max_rubber_uptake_abs:.4f} mmol/gDW/h")
        
        # B. 増殖時の最大フラックス (Growth 最大化時のゴム分解)
        model.objective = 'Growth'
        sol_max_growth = model.optimize()
        mu_max = sol_max_growth.objective_value
        rubber_uptake_at_mu_max = abs(sol_max_growth.fluxes['EX_rubber_bulk_e'])
        print(f"\nAt Maximum Growth (mu = {mu_max:.4f} 1/h):")
        print(f"Rubber Uptake Flux: {rubber_uptake_at_mu_max:.4f} mmol/gDW/h")

    # 2. 数学的積分の計算
    # 単位変換: ゴム(C5H8)の分子量 = 68.12 g/mol = 0.06812 g/mmol
    # 分解速度 (g/gDW/h) = q_rubber (mmol/gDW/h) * 0.06812
    q_rubber_g = rubber_uptake_at_mu_max * 0.06812
    
    # 微分方程式モデル:
    # dB/dt = mu_max * B
    # dR/dt = - q_rubber_g * B
    def system(y, t, mu, q):
        B, R = y
        dBdt = mu * B
        dRdt = -q * B
        return [dBdt, dRdt]
    
    t = np.linspace(0, 672, 673)
    y0 = [0.1, 100.0] # 初期バイオマス 0.1g/L, ゴム 100g/L
    
    sol = odeint(system, y0, t, args=(mu_max, q_rubber_g))
    biomass = sol[:, 0]
    rubber = sol[:, 1]
    
    # ゴムがマイナスにならないようにクリップ
    rubber = np.clip(rubber, 0, 100.0)
    
    # 50%分解 (50g/L到達) の時間を特定
    t_50 = np.argmax(rubber <= 50.0) if np.any(rubber <= 50.0) else -1
    t_100 = np.argmax(rubber <= 0.1) if np.any(rubber <= 0.1) else -1
    
    print("\n--- Theoretical Projection (672 hours) ---")
    print(f"Final Biomass: {biomass[-1]:.2e} g/L")
    print(f"Final Rubber: {rubber[-1]:.2f} g/L")
    print(f"Degradation Rate: {(100.0 - rubber[-1]) / 100.0 * 100:.2f}%")
    
    if t_50 > 0:
        print(f"Time to 50% degradation: {t_50} hours")
    else:
        print("Cannot reach 50% degradation within 672 hours.")
        
    if t_100 > 0:
        print(f"Time to 100% degradation: {t_100} hours")

if __name__ == "__main__":
    calculate_static_limits()
