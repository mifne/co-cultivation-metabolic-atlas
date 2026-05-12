import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")

for species in ["Actinoplanes_sp_OR16_lcp.xml", "Rhizobacter_gummiphilus_NS21.xml"]:
    print(f"\n--- {species} ---")
    model = cobra.io.read_sbml_model(str(sbml_dir / species))
    # Find biomass reaction
    for rxn in model.reactions:
        if 'biomass' in rxn.id.lower() or 'growth' in rxn.id.lower():
            print(f"Biomass RXN: {rxn.id}")
            # Print reactants that are not in my minimal medium
            standard_mets = ['glc__D_c', 'nh4_c', 'pi_c', 'so4_c', 'mg2_c', 'ca2_c', 'k_c', 'fe2_c', 'fe3_c', 'h_c', 'h2o_c', 'o2_c', 'atp_c', 'adp_c', 'nad_c', 'nadh_c', 'nadp_c', 'nadph_c']
            for met, coeff in rxn.metabolites.items():
                if coeff < 0: # reactant
                    # Strip M_ prefix if needed
                    mid = met.id
                    if mid.startswith('M_'): mid = mid[2:]
                    if mid not in standard_mets and 'coa' not in mid:
                        print(f"  Missing reactant: {met.id} ({met.name})")
