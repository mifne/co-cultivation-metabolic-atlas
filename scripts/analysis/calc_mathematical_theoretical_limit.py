import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
"""
Pure Mathematical Calculation of Theoretical Rubber Degradation Limits
FBAの最大フラックス値を用いた、純粋な数学的積分による分解ポテンシャルの計算
"""

import numpy as np
import matplotlib.pyplot as plt
import cobra
from pathlib import Path

def run_mathematical_projection():
    print("--- Pure Mathematical Theoretical Limit Analysis ---")
    
    # 1. OR16モデルの純粋な能力値を取得
    model_path = Path('models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml')
    model = cobra.io.read_sbml_model(str(model_path))
    
    with model:
        # 1. まず全ての炭素・エネルギー源となる取り込みを閉じる
        for r in model.exchanges:
            if r.lower_bound < 0:
                r.lower_bound = 0
                
        # 2. 必須無機塩類・酸素・水のみ無尽蔵に開く
        minimal_media = ['EX_h2o_e', 'EX_h_e', 'EX_nh4_e', 'EX_o2_e', 'EX_pi_e', 'EX_so4_e', 
                         'EX_ca2_e', 'EX_mg2_e', 'EX_fe3_e', 'EX_mn2_e', 'EX_zn2_e', 'EX_cu2_e', 
                         'EX_cobalt2_e', 'EX_cl_e', 'EX_k_e', 'EX_na_e', 'EX_ni2_e', 'EX_rubber_bulk_e']
        
        for r_id in minimal_media:
            try:
                model.reactions.get_by_id(r_id).lower_bound = -1000.0
            except KeyError:
                pass
        
        # ゴム分解反応の上限も開く
        model.reactions.get_by_id('R_LCP').upper_bound = 1000.0
        
        # 増殖を最大化した場合のゴム取り込み速度を計算
        model.objective = 'Growth'
        sol = model.optimize()
        
        mu_max = sol.objective_value
        q_rubber_mmol = abs(sol.fluxes.get('EX_rubber_bulk_e', 0.0))
        
        print(f"FBA Calculated Limits:")
        print(f"  Maximum Growth Rate (mu): {mu_max:.4f} 1/h")
        print(f"  Specific Rubber Uptake (q_rubber): {q_rubber_mmol:.6f} mmol/gDW/h")
        
        # 単位変換: C5H8 (MW = 68.12 g/mol)
        q_rubber_g = q_rubber_mmol * 0.06812
        print(f"  Specific Rubber Uptake (q_rubber): {q_rubber_g:.6f} g/gDW/h")

    # 2. 現実的な限界への補正
    # FBAの増殖速度 47.08 1/h は生物学的にあり得ない（大腸菌でも最大2.0程度）。
    # ここでは、放線菌の現実的な最大増殖速度 (例: 0.1 1/h) を上限とする。
    realistic_mu_max = min(mu_max, 0.1) 
    
    print("\n--- Projecting with Realistic Bounds ---")
    print(f"  Using Realistic mu: {realistic_mu_max:.4f} 1/h")
    
    # 3. オイラー法による積分計算 (672時間)
    dt = 0.1
    time_steps = int(672 / dt)
    
    time = np.zeros(time_steps)
    biomass = np.zeros(time_steps)
    rubber = np.zeros(time_steps)
    
    biomass[0] = 0.1  # 初期バイオマス 0.1 g/L
    rubber[0] = 100.0 # 初期ゴム 100 g/L
    
    degraded_50_time = -1
    
    for i in range(1, time_steps):
        t = i * dt
        
        # バイオマスの増殖 (ゴムがある場合のみ増殖)
        if rubber[i-1] > 0:
            dB = realistic_mu_max * biomass[i-1] * dt
        else:
            dB = 0
            
        biomass[i] = biomass[i-1] + dB
        
        # ゴムの分解
        dR = q_rubber_g * biomass[i-1] * dt
        rubber[i] = max(0.0, rubber[i-1] - dR)
        
        time[i] = t
        
        # 50% 到達判定
        if rubber[i] <= 50.0 and degraded_50_time == -1:
            degraded_50_time = t
            
    final_deg_rate = (100.0 - rubber[-1]) / 100.0 * 100
    
    print(f"\nFinal State after 672 hours:")
    print(f"  Biomass: {biomass[-1]:.2f} g/L")
    print(f"  Rubber Remaining: {rubber[-1]:.2f} g/L")
    print(f"  Degradation Rate: {final_deg_rate:.2f}%")
    if degraded_50_time > 0:
        print(f"  Reached 50% Degradation at: {degraded_50_time:.1f} hours")
    else:
        print("  Did not reach 50% Degradation.")

    # 4. プロット
    plt.figure(figsize=(10, 6))
    plt.plot(time, rubber, label='Rubber Concentration (g/L)')
    plt.axhline(50.0, color='r', linestyle=':', label='50% Target')
    if degraded_50_time > 0:
        plt.axvline(degraded_50_time, color='g', linestyle='--', label=f'50% reached at {degraded_50_time:.0f}h')
    
    plt.title(f'Theoretical Rubber Degradation Limit\n(q_rubber={q_rubber_g:.4f} g/gDW/h, mu_max={realistic_mu_max:.2f})')
    plt.xlabel('Time (hours)')
    plt.ylabel('Rubber Concentration (g/L)')
    plt.legend()
    plt.grid(True)
    plt.savefig('scripts/mathematical_theoretical_limit.png')
    print("📊 Saved projection plot to scripts/mathematical_theoretical_limit.png")

if __name__ == "__main__":
    run_mathematical_projection()
