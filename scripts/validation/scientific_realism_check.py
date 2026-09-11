import cobra
import numpy as np
from pathlib import Path
from src.utils import load_sbml_models, select_consortium_models, get_initial_params
from src.dfba_simulator import dFBASimulator

def validate_biological_realism():
    sbml_dir = Path('models/sbml/final_consortium')
    all_models = load_sbml_models(sbml_dir)
    models = select_consortium_models(all_models)
    
    print("\n" + "="*50)
    print("SCIENTIFIC VALIDATION: BIOLOGICAL REALISM")
    print("="*50)

    # テストシナリオ 1: 完全餓死状態 (炭素源ゼロ)
    print("\n[Scenario 1] Absolute Starvation (No Carbon Source)")
    initial_biomass, base_metabolites = get_initial_params(models)
    
    # 炭素源および有機微量要素を強制的にゼロにする (厳密な完全餓死)
    starvation_metabolites = base_metabolites.copy()
    inorganic_ions = [
        'nh4_e', 'pi_e', 'o2_e', 'so4_e', 'mg2_e', 'ca2_e', 'k_e', 'cl_e',
        'fe3_e', 'fe2_e', 'h_e', 'h2o_e', 'co2_e', 'zn2_e', 'mn2_e', 'cu2_e',
        'cu_e', 'cobalt2_e', 'ni2_e', 'mobd_e', 'ppi_e', 'tsul_e', 'salchs4fe_e'
    ]
    for met in list(starvation_metabolites.keys()):
        if met not in inorganic_ions:
            starvation_metabolites[met] = 0.0
            
    sim_starve = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=starvation_metabolites,
        initial_rubber=0.0,
        dt=1.0
    )
    
    # 5時間経過させて、培地中の微量アミノ酸(0.01mM)を食い尽くさせる
    for _ in range(5):
        state_starve = sim_starve.step({}, {})
        
    for name, s in state_starve.species.items():
        print(f"  - {name:30}: μ = {s.growth_rate:.6f} h-1")
        # 0.02 以下なら生物学的に「維持のみ（または極めて遅い独立栄養的成長）」とみなせる
        if s.growth_rate > 0.02:
            print(f"    ⚠️  WARNING: Unrealistic growth detected! ({s.growth_rate})")
        else:
            print(f"    ✅ Realistic (Minimal growth)")

    # テストシナリオ 2: ゴムのみを炭素源とした成長
    print("\n[Scenario 2] Growth on Rubber (Natural Synergy)")
    sim_rubber = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=starvation_metabolites,
        initial_rubber=100.0, # ゴムあり
        dt=1.0
    )
    
    # ゴム分解を10時間進めて中間体を蓄積させる
    print("  Simulating 10 hours of degradation...")
    for _ in range(10):
        state_rubber = sim_rubber.step({'Actinoplanes_sp_OR16_lcp': 0.1}, {})
    
    for name, s in state_rubber.species.items():
        print(f"  - {name:30}: μ = {s.growth_rate:.6f} h-1")
        if s.growth_rate > 0.01:
            print(f"    ✅ Realistic (Growth observed on carbon source)")
        elif 'Lactobacillus' in name:
             print(f"    ℹ️  LP (Expected low growth without specialized sugars)")
        else:
            print(f"    ⚠️  WARNING: Low growth on rubber source!")

    # テストシナリオ 3: ソルバーの安定性 (Infeasibility Check)
    print("\n[Scenario 3] Solver Feasibility Check")
    # 背景栄養素をさらに絞った時にソルバーが壊れないか確認
    # (もし 0.001 が厳しすぎると、ここでソルバーが None を返す)
    for name in models.keys():
        solution = sim_rubber.solve_fba(name)
        if solution:
            print(f"  - {name:30}: Solver OK (Status: {solution.status})")
        else:
            print(f"  - {name:30}: ❌ Solver FAILED (Infeasible)")

if __name__ == "__main__":
    validate_biological_realism()
