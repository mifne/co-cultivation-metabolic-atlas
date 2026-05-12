import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Rhizobacter_gummiphilus_NS21.xml"))

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
    
    # Try growing on C30_oligo (product of OR16 LCP)
    try:
        model.reactions.EX_C30_oligo_e.lower_bound = -10
        sol = model.optimize()
        print(f"Growth on C30_oligo: {sol.objective_value}")
    except:
        print("EX_C30_oligo_e not found")

    # Try growing on rubber_fragment_e
    try:
        model.reactions.EX_rubber_fragment_e.lower_bound = -10
        sol2 = model.optimize()
        print(f"Growth on Rubber Fragment: {sol2.objective_value}")
    except:
        print("EX_rubber_fragment_e not found")

    # Try growing on odtd_e
    try:
        model.reactions.EX_odtd_e.lower_bound = -10
        sol3 = model.optimize()
        print(f"Growth on ODTD: {sol3.objective_value}")
    except:
        print("EX_odtd_e not found")

