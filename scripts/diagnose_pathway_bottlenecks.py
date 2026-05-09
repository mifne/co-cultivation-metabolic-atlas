import cobra
import numpy as np
from pathlib import Path
from main import load_sbml_models, select_consortium_models, get_initial_params
from src.dfba_simulator import dFBASimulator

def diagnose_bottlenecks():
    print("🚀 Starting Metabolic Pathway Audit...")
    sbml_dir = Path('models/sbml/final_consortium')
    all_models = load_sbml_models(sbml_dir)
    models = select_consortium_models(all_models)
    initial_biomass, initial_metabolites = get_initial_params(models)
    
    # RL環境の典型的な低栄養状態を模倣 (グルコースなし、低NH4)
    test_metabolites = initial_metabolites.copy()
    test_metabolites['glc__D_e'] = 0.0
    test_metabolites['nh4_e'] = 1.0 # 少量の窒素
    test_metabolites['o2_e'] = 0.25 # 飽和
    
    sim = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=test_metabolites,
        initial_rubber=100.0,
        volume=1.0,
        dt=0.2
    )

    # 培地の全代謝物を表示
    print("\n[Initial Metabolites Pool]")
    for met, conc in test_metabolites.items():
        if conc > 0:
            print(f"  {met:20}: {conc:.4f}")

    for species_name, model in models.items():
        print(f"\n--- Diagnosing {species_name} ---")
        sim.set_uptake_constraints(species_name, test_metabolites, dynamic_kla=200.0)
        
        # NS21のゴム取り込みを強制開放（デバッグ用）
        if 'NS21' in species_name:
            if 'EX_rubber_bulk_e' in model.reactions:
                model.reactions.get_by_id('EX_rubber_bulk_e').lower_bound = -1000.0
            if 'EX_C30_oligo_e' in model.reactions:
                model.reactions.get_by_id('EX_C30_oligo_e').lower_bound = -1000.0

        # 1. 現状のFBA
        sol = model.optimize()
        print(f"  FBA Status: {sol.status}")
        if sol.status == 'optimal':
            print(f"  Biomass Flux: {sol.objective_value:.6f}")
            print(f"  [Significant Fluxes (>0.1)]")
            for rid, flux in sol.fluxes.items():
                if abs(flux) > 0.1 and ('EX_' in rid or 'R_' in rid):
                    print(f"    {rid:20}: {flux:9.4f}")
        
        # 2. ターゲット反応の詳細
        targets = []
        if 'OR16' in species_name:
            targets = ['R_LCP', 'R_C30t', 'R_C30_cat', 'EX_C30_oligo_e', 'EX_rubber_bulk_e', 'EX_o2_e']
        elif 'NS21' in species_name:
            targets = ['R_ROXB', 'R_ROXA', 'R_ODTDt', 'R_ODTD_cat', 'EX_odtd_e', 'EX_C30_oligo_e', 'EX_rubber_bulk_e', 'EX_o2_e']

        print(f"  [Reaction Audit]")
        for rid in targets:
            if rid in model.reactions:
                rxn = model.reactions.get_by_id(rid)
                flux = sol.fluxes[rid] if sol.status == 'optimal' else 0.0
                print(f"    {rid:16}: Flux={flux:9.4f}, LB={rxn.lower_bound:9.4f}, UB={rxn.upper_bound:9.4f}")
        
    # --- 究極の診断: アミノ酸も糖もなし、ゴムのみ ---
    print("\n[Ultimate Test: Rubber as SOLE Carbon Source]")
    starved_metabolites = test_metabolites.copy()
    for aa in ['ala__L_e', 'arg__L_e', 'asn__L_e', 'asp__L_e', 'cys__L_e', 'gln__L_e', 'glu__L_e', 'gly_e', 'his__L_e', 'ile__L_e', 'leu__L_e', 'lys__L_e', 'met__L_e', 'phe__L_e', 'pro__L_e', 'ser__L_e', 'thr__L_e', 'trp__L_e', 'tyr__L_e', 'val__L_e']:
        starved_metabolites[aa] = 0.0
    
    for species_name, model in models.items():
        print(f"\n--- Diagnosing {species_name} (Starved) ---")
        sim.set_uptake_constraints(species_name, starved_metabolites, dynamic_kla=200.0)
        
        # ゴム取り込みを強制開放
        if 'rubber_e' in sim.exchange_reactions[species_name]:
            rxn_id = sim.exchange_reactions[species_name]['rubber_e']
            model.reactions.get_by_id(rxn_id).lower_bound = -1000.0
            print(f"  Opened {rxn_id} (rubber_e) for {species_name}")

        sol = model.optimize()
        print(f"  FBA Status: {sol.status}")
        if sol.status == 'optimal':
            print(f"  Biomass Flux: {sol.objective_value:.6f}")
            if sol.objective_value < 1e-6:
                # なぜ増殖できないのか？
                # エネルギー源としてのゴム分解をチェック
                with model:
                    model.objective = 'R_LCP' if 'OR16' in species_name else 'R_ROXB'
                    sol_deg = model.optimize()
                    print(f"  Degradation possible?: {sol_deg.status} (Flux: {sol_deg.objective_value:.6f})")
                    
                    # 必須アミノ酸や補酵素の不足をチェック
                    # (アミノ酸を抜いたので、自力で合成できないと詰む)
                    if sol_deg.objective_value > 0:
                        print("  Degradation is possible but no growth. Likely missing AA synthesis pathways or vitamins.")
        else:
            print("  INFEASIBLE on rubber alone.")
            # 緩和テスト
            sim._diagnose_infeasibility(species_name, model)

if __name__ == "__main__":
    diagnose_bottlenecks()
