import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

with model:
    # Open standard medium
    for m in ['o2_e', 'nh4_e', 'pi_e', 'so4_e', 'mg2_e', 'ca2_e', 'k_e', 'fe2_e', 'fe3_e', 'h_e', 'h2o_e']:
        try: model.reactions.get_by_id(f"R_EX_{m[2:]}").lower_bound = -1000
        except:
            try: model.reactions.get_by_id(f"EX_{m[2:]}").lower_bound = -1000
            except: pass
    
    # Open glucose
    model.reactions.R_EX_glc__D_e.lower_bound = -10
    sol = model.optimize()
    print(f"Growth on Glucose: {sol.objective_value}")
    
    if sol.status == 'optimal':
        # Find where accoa_c comes from in glucose growth
        accoa = model.metabolites.accoa_c
        print(f"\n--- Reactions involving {accoa.id} ---")
        for r in accoa.reactions:
            if abs(sol.fluxes[r.id]) > 1e-6:
                print(f"Flux {sol.fluxes[r.id]:.4f}: {r.id} ({r.reaction})")

