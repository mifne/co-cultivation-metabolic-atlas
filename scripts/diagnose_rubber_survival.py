import cobra
import pandas as pd
from pathlib import Path
from main import load_sbml_models, select_consortium_models, get_initial_params
from src.dfba_simulator import dFBASimulator

def diagnose_rubber_survival():
    print("🚀 Starting Deep Metabolic Audit: Rubber Survival Test...")
    sbml_dir = Path('models/sbml/final_consortium')
    all_models = load_sbml_models(sbml_dir)
    models = select_consortium_models(all_models)
    
    results = []

    for species_name, model in models.items():
        if 'Lactobacillus' in species_name: continue
        
        print(f"\n--- Species: {species_name} ---")
        
        # 基本の供給 (main.py から全サプリメントを持演)
        m9_mets = [
            'pi_e', 'so4_e', 'mg2_e', 'ca2_e', 'k_e', 'cl_e', 'fe3_e', 'fe2_e', 'h_e', 'h2o_e', 'co2_e', 'nh4_e',
            'nac_e', 'ribflv_e', 'pnto__R_e', 'thm_e', 'btn_e', '4abz_e', 'fol_e', 'nicnt_e', 'ade_e', 'gua_e', 'ura_e', 'xan_e', 'orot_e', 'ins_e', 'thymd_e'
        ]
        
        def reset_medium():
            medium = model.medium
            for rxn in model.exchanges:
                medium[rxn.id] = 0.0
            for met in m9_mets:
                for rxn in model.exchanges:
                    if met in [m.id.replace('M_', '') for m in rxn.metabolites]:
                        medium[rxn.id] = 1000.0
            # 酸素供給
            for rxn in model.exchanges:
                if 'o2_e' in [m.id.replace('M_', '') for m in rxn.metabolites]:
                    medium[rxn.id] = 1000.0
            model.medium = medium

        # simulatorのインスタンス化
        initial_biomass, initial_metabolites = get_initial_params(models)
        sim = dFBASimulator(models, initial_biomass, initial_metabolites)
        YE_COMPONENTS = sim.YE_COMPONENTS
        
        with model:
            print(f"  Available exchanges for {species_name}: {list(sim.exchange_reactions[species_name].keys())[:10]}...")
            # 1. YEによる微量要素の解放 (Simulatorのロジックを忠実に再現)
            count = 0
            for met_id in YE_COMPONENTS:
                if met_id in sim.exchange_reactions[species_name]:
                    rxn_id = sim.exchange_reactions[species_name][met_id]
                    model.reactions.get_by_id(rxn_id).lower_bound = -1.0
                    count += 1
            print(f"  Unlocked {count} YE components for {species_name}")
            
            # 2. ゴムの供給
            rubber_rxn = sim.exchange_reactions[species_name].get('rubber_e')
            if rubber_rxn:
                model.reactions.get_by_id(rubber_rxn).lower_bound = -10.0
                sol_ye = model.optimize()
                print(f"  Case 0 (YE Package + Rubber): Biomass={sol_ye.objective_value:.6f}, Status={sol_ye.status}")
                if sol_ye.status == 'optimal' and sol_ye.objective_value > 0.001:
                    print(f"    ✅ SUCCESS")
                else:
                    print(f"    ❌ FAILURE. Trying to open ALL exchanges...")
                    for rxn in model.exchanges: rxn.lower_bound = -1000.0
                    sol_all = model.optimize()
                    print(f"    Biomass with ALL open: {sol_all.objective_value:.6f}")
                    if sol_all.objective_value > 0:
                        from cobra.medium import minimal_medium
                        print(f"    New minimal medium: {minimal_medium(model, 0.1)}")

        # Case 1: Rubber as Sole Carbon Source
        reset_medium()
        rubber_rxn = None
        for rid in ['EX_rubber_bulk_e', 'R_EX_rubber_bulk_e']:
            if rid in model.reactions:
                rubber_rxn = rid
                break
        
        if rubber_rxn:
            model.reactions.get_by_id(rubber_rxn).lower_bound = -10.0
            sol = model.optimize()
            print(f"  Case 1 (Rubber Only): Biomass={sol.objective_value:.6f}, Status={sol.status}")
            results.append({'Species': species_name, 'Case': 'Rubber Only', 'Biomass': sol.objective_value, 'Status': sol.status})
            
            if sol.objective_value < 1e-6:
                # 必須アミノ酸がないとダメなのか？
                with model:
                    for aa_rxn in model.exchanges:
                        if '__L_e' in aa_rxn.id:
                            aa_rxn.lower_bound = -0.1
                    sol_aa = model.optimize()
                    print(f"  Case 2 (Rubber + 0.1mM AA): Biomass={sol_aa.objective_value:.6f}, Status={sol_aa.status}")
                    results.append({'Species': species_name, 'Case': 'Rubber + 0.1mM AA', 'Biomass': sol_aa.objective_value, 'Status': sol_aa.status})
                    
                    if sol_aa.objective_value > 0.001:
                        # どのAAが必須なのか？ (Single omission test)
                        print("    Identifying essential amino acids...")
                        essential_aas = []
                        for aa_rxn in model.exchanges:
                            if '__L_e' in aa_rxn.id:
                                with model:
                                    # 全てのAAを供給した状態で、一つだけ抜く
                                    for other_aa in model.exchanges:
                                        if '__L_e' in other_aa.id: other_aa.lower_bound = -0.1
                                    aa_rxn.lower_bound = 0.0
                                    sol_omit = model.optimize()
                                    if sol_omit.objective_value < 1e-6:
                                        essential_aas.append(aa_rxn.id)
                        print(f"    Essential AAs: {essential_aas}")

        # Case 3: O2 Dependency of LCP/ROX
        if 'OR16' in species_name:
            target = 'R_LCP'
        else:
            target = 'R_ROXB'
            
        if target in model.reactions:
            with model:
                # 増殖を目的関数にして、酸素を徐々に減らす
                o2_rxn = None
                for rxn in model.exchanges:
                    if 'o2_e' in [m.id.replace('M_', '') for m in rxn.metabolites]:
                        o2_rxn = rxn
                        break
                
                print(f"  Checking {target} O2 sensitivity:")
                for o2_limit in [0.1, 1.0, 5.0, 10.0]:
                    o2_rxn.lower_bound = -o2_limit
                    # AAは供給した状態にする
                    for aa_rxn in model.exchanges:
                        if '__L_e' in aa_rxn.id: aa_rxn.lower_bound = -0.1
                    model.reactions.get_by_id(rubber_rxn).lower_bound = -10.0
                    sol_o2 = model.optimize()
                    print(f"    O2 Limit {o2_limit:4.1f}: {target}={sol_o2.fluxes[target]:7.4f}, Biomass={sol_o2.objective_value:7.4f}")

    df = pd.DataFrame(results)
    print("\n[Summary of Results]")
    print(df)

if __name__ == "__main__":
    diagnose_rubber_survival()
