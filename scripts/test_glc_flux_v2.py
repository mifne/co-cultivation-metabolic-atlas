import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

with model:
    # 1. Close all carbon
    for r in model.exchanges:
        if r.lower_bound < 0: r.lower_bound = 0
    # 2. Open standard medium
    for m in ['o2_e', 'nh4_e', 'pi_e', 'so4_e', 'mg2_e', 'ca2_e', 'k_e', 'fe2_e', 'fe3_e', 'h_e', 'h2o_e']:
        try: model.reactions.get_by_id(f"R_EX_{m[2:]}").lower_bound = -1000
        except:
            try: model.reactions.get_by_id(f"EX_{m[2:]}").lower_bound = -1000
            except: pass
    
    # 3. Find glucose exchange
    glc_rxn = None
    for r in model.exchanges:
        if 'glc__D_e' in r.reaction:
            glc_rxn = r
            break
    
    if glc_rxn:
        print(f"Glucose RXN: {glc_rxn.id}")
        glc_rxn.lower_bound = -10
        sol = model.optimize()
        print(f"Growth on Glucose: {sol.objective_value}")
        
        if sol.status == 'optimal' and sol.objective_value > 1e-6:
            # Find where accoa_c is used
            if 'accoa_c' in model.metabolites:
                accoa = model.metabolites.accoa_c
                print(f"\n--- Active Reactions involving {accoa.id} ---")
                for r in accoa.reactions:
                    if abs(sol.fluxes[r.id]) > 1e-6:
                        print(f"Flux {sol.fluxes[r.id]:.4f}: {r.id} ({r.reaction})")
    else:
        print("Glucose exchange not found")

