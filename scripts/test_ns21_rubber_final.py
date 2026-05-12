import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Rhizobacter_gummiphilus_NS21.xml"))

# Identify actual current bounds
essentials = ['EX_ca2_e', 'EX_cl_e', 'EX_cobalt2_e', 'EX_cu2_e', 'EX_istfrnB_e', 'EX_k_e', 'EX_mg2_e', 'EX_mn2_e', 'EX_o2_e', 'EX_so4_e', 'EX_tyrp_e', 'EX_zn2_e']

with model:
    for r in model.exchanges: r.bounds = (0, 1000)
    for rid in essentials:
        if rid in model.reactions: model.reactions.get_by_id(rid).bounds = (-1000, 1000)
    
    # Try C30 (from OR16)
    model.reactions.EX_C30_oligo_e.lower_bound = -10
    sol = model.optimize()
    print(f"NS21 C30 Growth: {sol.objective_value}")

