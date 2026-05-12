import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Rhizobacter_gummiphilus_NS21.xml"))

with model:
    # Close all carbon
    for r in model.exchanges: r.lower_bound = 0
    # Open essential medium
    essentials = ['EX_ca2_e', 'EX_cl_e', 'EX_cobalt2_e', 'EX_cu2_e', 'EX_istfrnB_e', 'EX_k_e', 'EX_mg2_e', 'EX_mn2_e', 'EX_o2_e', 'EX_so4_e', 'EX_tyrp_e', 'EX_zn2_e', 'EX_h2o_e', 'EX_h_e', 'EX_pi_e']
    for rid in essentials:
        if rid in model.reactions: model.reactions.get_by_id(rid).lower_bound = -1000
    
    # Try ODTD as carbon source
    try:
        model.reactions.EX_odtd_e.lower_bound = -10
    except KeyError:
        print("EX_odtd_e not found")

    # Set objective to PHA
    model.objective = 'EX_pha_c'
    sol = model.optimize()
    print(f"PHA production on ODTD: {sol.objective_value}")

