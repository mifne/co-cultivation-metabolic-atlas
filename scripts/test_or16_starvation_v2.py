import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

# Simulating the exact state from debug_simulator_v6
# Reaction EX_arg__L_e: LB=-0.01
# Reaction EX_h2o_e: LB=-1000.0
# Reaction EX_h_e: LB=-1000.0
# Reaction EX_leu__L_e: LB=-0.01
# Reaction EX_o2_e: LB=-1000.0
# Reaction EX_trp__L_e: LB=-0.01
# Reaction EX_rubber_bulk_e: LB=-19.8

with model:
    for r in model.exchanges: r.bounds = (0, 1000)
    model.reactions.EX_arg__L_e.lower_bound = -0.01
    model.reactions.EX_h2o_e.lower_bound = -1000.0
    model.reactions.EX_h_e.lower_bound = -1000.0
    model.reactions.EX_leu__L_e.lower_bound = -0.01
    model.reactions.EX_o2_e.lower_bound = -1000.0
    model.reactions.EX_trp__L_e.lower_bound = -0.01
    model.reactions.EX_rubber_bulk_e.lower_bound = -19.8
    
    sol = model.optimize()
    print(f"Growth in starvation (with 0.01 AA but NO micronutrients): {sol.objective_value}")
    
    # Add micronutrients
    essentials = ['EX_ca2_e', 'EX_cl_e', 'EX_cobalt2_e', 'EX_cu_e', 'EX_k_e', 'EX_mg2_e', 'EX_mn2_e', 'EX_salchs4fe_e', 'EX_tsul_e', 'EX_ump_e', 'EX_xtsn_e', 'EX_zn2_e']
    for rid in essentials: model.reactions.get_by_id(rid).lower_bound = -1000
    
    sol2 = model.optimize()
    print(f"Growth in starvation (WITH micronutrients): {sol2.objective_value}")

