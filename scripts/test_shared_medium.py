import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")

models = {
    'OR16': cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml")),
    'NS21': cobra.io.read_sbml_model(str(sbml_dir / "Rhizobacter_gummiphilus_NS21.xml"))
}

# OR16 needs: ca2, cl, cobalt2, cu, k, mg, mn, o2, salchs4fe, tsul, ump, xtsn, zn
# NS21 needs: ca2, cl, cobalt2, cu2, istfrnB, k, mg, mn, o2, so4, tyrp, zn

shared_medium = [
    'EX_ca2_e', 'EX_cl_e', 'EX_cobalt2_e', 'EX_cu_e', 'EX_cu2_e', 'EX_k_e', 'EX_mg2_e', 'EX_mn2_e', 
    'EX_o2_e', 'EX_so4_e', 'EX_zn2_e',
    'EX_salchs4fe_e', 'EX_tsul_e', 'EX_ump_e', 'EX_xtsn_e', # for OR16
    'EX_istfrnB_e', 'EX_tyrp_e' # for NS21
]

for name, model in models.items():
    print(f"\n--- Testing {name} with shared medium ---")
    with model:
        for r in model.exchanges: r.lower_bound = 0
        for rid in shared_medium:
            if rid in model.reactions: model.reactions.get_by_id(rid).lower_bound = -1000
        
        # Test Carbon
        model.reactions.EX_glc__D_e.lower_bound = -10
        print(f"Growth on Glucose: {model.optimize().objective_value}")
        
        # Test Rubber
        if name == 'OR16':
            model.reactions.EX_glc__D_e.lower_bound = 0
            model.reactions.EX_rubber_bulk_e.lower_bound = -10
            print(f"Growth on Rubber: {model.optimize().objective_value}")
