import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

with model:
    for r in model.exchanges: r.lower_bound = 0
    # USE EXACT REACTION IDS IDENTIFIED
    essentials = ['EX_ca2_e', 'EX_cl_e', 'EX_cobalt2_e', 'EX_cu_e', 'EX_k_e', 'EX_mg2_e', 'EX_lmn2_e', 'EX_co2_e', 'EX_salchs4fe_e', 'EX_tsul_e', 'EX_23cump_e', 'EX_xtsn_e', 'EX_zn2_e']
    for rid in essentials:
        model.reactions.get_by_id(rid).lower_bound = -1000
    
    # Try Glucose
    model.reactions.EX_glc__D_e.lower_bound = -10
    print(f"Growth with Glucose: {model.optimize().objective_value}")
    
    # Try Rubber
    model.reactions.EX_glc__D_e.lower_bound = 0
    model.reactions.EX_rubber_bulk_e.lower_bound = -10
    print(f"Growth with Rubber: {model.optimize().objective_value}")

