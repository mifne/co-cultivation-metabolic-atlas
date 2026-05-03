from cobra.io import read_sbml_model

model = read_sbml_model('models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml')
# set minimal media + fragment
media = {
    'EX_o2_e': 1000.0,
    'EX_pi_e': 1000.0,
    'EX_so4_e': 1000.0,
    'EX_ca2_e': 1000.0,
    'EX_mg2_e': 1000.0,
    'EX_fe3_e': 1000.0,
    'EX_k_e': 1000.0,
    'EX_cl_e': 1000.0,
    'EX_h2o_e': 1000.0,
    'EX_rubber_fragment_e': 10.0,
    'EX_nh4_e': 0.0,  # No nitrogen
}
for rxn in model.reactions:
    if rxn.id.startswith('EX_'):
        if rxn.id in media:
            rxn.lower_bound = -media[rxn.id]
        else:
            rxn.lower_bound = 0.0

sol = model.optimize()
print(f"Status: {sol.status}, Objective: {sol.objective_value}")
print(f"PHA Flux: {sol.fluxes.get('EX_pha_c', 0.0)}")
print(f"Biomass: {sol.fluxes.get('BIOMASS_NS21_core', 0.0)}")
