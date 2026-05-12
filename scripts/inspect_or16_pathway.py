import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

# Check R_R_C30_cat
rxn = model.reactions.get_by_id("R_R_C30_cat")
print(f"Reaction: {rxn.id}")
print(f"Metabolites: {rxn.reaction}")

# Check if products of C30 cat (accoa, ppcoa) can reach Biomass
print("\n--- Checking connectivity to growth ---")
with model:
    # Close all carbon
    for r in model.exchanges:
        if r.lower_bound < 0: r.lower_bound = 0
    # Open necessary
    for m in ['o2_e', 'nh4_e', 'pi_e', 'so4_e', 'mg2_e', 'ca2_e', 'k_e', 'fe2_e', 'fe3_e', 'h_e', 'h2o_e']:
        try: model.reactions.get_by_id(f"R_EX_{m[2:]}").lower_bound = -1000
        except: pass
    
    # Try growing on C30_oligo directly
    try:
        model.reactions.R_EX_C30_oligo_e.lower_bound = -10
        sol = model.optimize()
        print(f"Growth on C30_oligo: {sol.objective_value}")
    except:
        print("R_EX_C30_oligo_e not found or error")

    # Try growing on Acetyl-CoA (internal)
    # Since we can't easily add internal source, let's check if accoa_c is a metabolite
    if 'M_accoa_c' in model.metabolites:
        print("accoa_c found")
        # Add a temporary demand reaction for accoa_c to see if it's connected
        try:
            dm = model.add_boundary(model.metabolites.M_accoa_c, type='sink')
            model.objective = dm
            sol = model.optimize()
            print(f"Connectivity to accoa_c: {sol.objective_value}")
        except: pass

