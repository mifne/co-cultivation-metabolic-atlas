import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

with model:
    # Close all carbon
    for r in model.exchanges:
        if r.lower_bound < 0: r.lower_bound = 0
    # Open necessary
    for m in ['o2_e', 'nh4_e', 'pi_e', 'so4_e', 'mg2_e', 'ca2_e', 'k_e', 'fe2_e', 'fe3_e', 'h_e', 'h2o_e']:
        try: model.reactions.get_by_id(f"R_EX_{m[2:]}").lower_bound = -1000
        except:
            try: model.reactions.get_by_id(f"EX_{m[2:]}").lower_bound = -1000
            except: pass
    
    # Try growing on C30_oligo directly
    model.reactions.EX_C30_oligo_e.lower_bound = -10
    sol = model.optimize()
    print(f"Growth on C30_oligo: {sol.objective_value}")

    # Try growing on rubber directly
    model.reactions.EX_C30_oligo_e.lower_bound = 0
    model.reactions.EX_rubber_bulk_e.lower_bound = -10
    sol2 = model.optimize()
    print(f"Growth on Rubber: {sol2.objective_value}")

