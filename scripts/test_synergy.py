import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")

models = {
    'OR16': cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml")),
    'NS21': cobra.io.read_sbml_model(str(sbml_dir / "Rhizobacter_gummiphilus_NS21.xml"))
}

shared_medium = [
    'EX_ca2_e', 'EX_cl_e', 'EX_cobalt2_e', 'EX_cu_e', 'EX_cu2_e', 'EX_k_e', 'EX_mg2_e', 'EX_mn2_e', 
    'EX_o2_e', 'EX_so4_e', 'EX_zn2_e',
    'EX_salchs4fe_e', 'EX_tsul_e', 'EX_ump_e', 'EX_xtsn_e',
    'EX_istfrnB_e', 'EX_tyrp_e'
]

print("--- Testing NS21 growth on OR16 products ---")
with models['NS21'] as ns21:
    for r in ns21.exchanges: ns21.reactions.get_by_id(r.id).lower_bound = 0
    for rid in shared_medium:
        if rid in ns21.reactions: ns21.reactions.get_by_id(rid).lower_bound = -1000
    
    # NS21 needs EX_C30_oligo_e
    ns21.reactions.EX_C30_oligo_e.lower_bound = -10
    print(f"NS21 growth on C30_oligo: {ns21.optimize().objective_value}")

