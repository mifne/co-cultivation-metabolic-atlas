"""
Ideal dFBA Simulation for Rubber Degradation Limit
理想環境下での動的シミュレーションによるゴム分解限界の検証
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from main import load_sbml_models, select_consortium_models, get_initial_params
from src.dfba_simulator import dFBASimulator

def run_ideal_simulation(scenario: str, max_time: int = 672):
    sbml_dir = Path('models/sbml/final_consortium')
    all_models = load_sbml_models(sbml_dir)
    
    if scenario == 'single':
        models = {k: v for k, v in all_models.items() if 'OR16' in k}
    else:
        models = select_consortium_models(all_models)
        
    initial_biomass, initial_metabolites = get_initial_params(models)
    
    # 理想環境: 栄養を無尽蔵に与える
    initial_metabolites['glc__D_e'] = 0.0 
    initial_metabolites['nh4_e'] = 100.0
    initial_metabolites['o2_e'] = 100.0

    sim = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=100.0,
        volume=1.0,
        dt=1.0,
        carrying_capacity=20.0 # Standard capacity
    )
    
    history = []
    print(f"\n🚀 Running ideal {scenario} simulation for {max_time} hours...")
    
    for t in range(max_time):
        # 毎ステップ「神の手」で最適環境を維持 (pH=7.0固定)
        sim.state.metabolites['h_e'] = 0.0001
        ratio = 10**(7.0 - sim.pKa)
        sim.buffer_base = sim.buffer_total * ratio / (1 + ratio)
        sim.buffer_acid = sim.buffer_total - sim.buffer_base

        # 栄養の補充
        sim.state.metabolites['glc__D_e'] = 0.0
        sim.state.metabolites['nh4_e'] = 100.0
        sim.state.metabolites['o2_e'] = 100.0
        for met in ['pi_e', 'so4_e', 'ca2_e', 'mg2_e', 'fe3_e', 'mn2_e', 'zn2_e', 'cu2_e', 'cobalt2_e', 'cl_e', 'k_e', 'na_e', 'ni2_e']:
            sim.state.metabolites[met] = 100.0

        # アミノ酸を制限
        for aa in ['arg__L_e', 'glu__L_e', 'ala__L_e', 'asn__L_e', 'asp__L_e', 'cys__L_e', 'gln__L_e', 'gly_e', 'his__L_e', 'ile__L_e', 'leu__L_e', 'lys__L_e', 'met__L_e', 'phe__L_e', 'pro__L_e', 'ser__L_e', 'thr__L_e', 'trp__L_e', 'tyr__L_e', 'val__L_e']:
             sim.state.metabolites[aa] = 0.0

        # シナジーの検証: NS21がバイオサーファクタントを供給していると仮定 (定常生産)
        if scenario == 'consortium':
            sim.state.metabolites['biosurfactant_e'] = 1.0 # Constant boost
            # 競合を避けるため、他の菌種は最小限に維持
            for name, state in sim.state.species.items():
                if 'NS21' in name or 'Lactobacillus' in name:
                    state.biomass = 0.1 

        sim.step({}, {})
        
        record = {
            'time': sim.state.time,
            'rubber': sim.state.rubber_concentration,
            'biomass_or16': sim.state.species.get([k for k in models.keys() if 'OR16' in k][0]).biomass,
        }
        history.append(record)
        if sim.state.rubber_concentration <= 0.01:
             print(f"  -> Rubber completely degraded at hour {sim.state.time}")
             break
             
    df = pd.DataFrame(history)
    return df

def main():
    df_single = run_ideal_simulation('single')
    df_cons = run_ideal_simulation('consortium')
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_single['time'], df_single['rubber'], label='OR16 Alone', linewidth=2)
    plt.plot(df_cons['time'], df_cons['rubber'], label='Consortium (Synergy)', linewidth=2, linestyle='--')
    plt.title('Theoretical Rubber Degradation: Single vs Consortium')
    plt.xlabel('Time (h)')
    plt.ylabel('Rubber Concentration (g/L)')
    plt.legend()
    plt.grid(True)
    plt.savefig('scripts/theoretical_degradation_limit.png')
    
    print(f"\nSingle Final Rubber: {df_single['rubber'].iloc[-1]:.2f} g/L")
    print(f"Consortium Final Rubber: {df_cons['rubber'].iloc[-1]:.2f} g/L")

if __name__ == '__main__':
    main()
