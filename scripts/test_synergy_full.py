import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")

models = {
    'OR16': cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml")),
    'NS21': cobra.io.read_sbml_model(str(sbml_dir / "Rhizobacter_gummiphilus_NS21.xml"))
}

ye_list = ['ca2_e', 'cl_e', 'cobalt2_e', 'cu_e', 'cu2_e', 'k_e', 'mg2_e', 'mn2_e', 'pnto__R_e', 'so4_e', 'thm_e', 'zn2_e', 'salchs4fe_e', 'tsul_e', 'ump_e', 'xtsn_e', 'istfrnB_e', 'tyrp_e', 'glu__L_e', 'leu__L_e', 'nmn_e', 'ppi_e', 'ura_e']

for name, m in models.items():
    print(f"\n--- Testing {name} Synergy ---")
    with m:
        for r in m.exchanges: r.bounds = (0, 1000)
        # Identify YE rxns for this model
        for comp in ye_list:
            # Match comp ID to reaction
            for r in m.exchanges:
                if comp in r.reaction:
                    r.bounds = (-1000, 1000)
        
        # Open Standard (O2, Pi, NH4)
        for std in ['co2_e', 'o2_e', 'pi_e', 'nh4_e', 'h_e', 'h2o_e']:
            for r in m.exchanges:
                if std in r.reaction: r.bounds = (-1000, 1000)
        
        if name == 'OR16':
            m.reactions.EX_rubber_bulk_e.lower_bound = -10
            print(f"OR16 Growth on Rubber: {m.optimize().objective_value}")
        else:
            m.reactions.EX_C30_oligo_e.lower_bound = -10
            print(f"NS21 Growth on C30: {m.optimize().objective_value}")
