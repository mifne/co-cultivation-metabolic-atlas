import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

# Identify actual current bounds
essentials = ['EX_ca2_e', 'EX_cl_e', 'EX_cobalt2_e', 'EX_cu_e', 'EX_k_e', 'EX_mg2_e', 'EX_mn2_e', 'EX_o2_e', 'EX_salchs4fe_e', 'EX_tsul_e', 'EX_ump_e', 'EX_xtsn_e', 'EX_zn2_e']

with model:
    for r in model.exchanges: r.bounds = (0, 1000)
    for rid in essentials:
        if rid in model.reactions: model.reactions.get_by_id(rid).bounds = (-1000, 1000)
    
    # Try Rubber
    model.reactions.EX_rubber_bulk_e.lower_bound = -10
    sol = model.optimize()
    print(f"OR16 Rubber Growth: {sol.objective_value}")

