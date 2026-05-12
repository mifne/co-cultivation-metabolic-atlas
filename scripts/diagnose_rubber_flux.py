import cobra
import numpy as np
from pathlib import Path
from src.dfba_simulator import dFBASimulator

# Load models
sbml_dir = Path("models/sbml/final_consortium")
models = {
    'OR16': cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml")),
    'NS21': cobra.io.read_sbml_model(str(sbml_dir / "Rhizobacter_gummiphilus_NS21.xml")),
    'LP': cobra.io.read_sbml_model(str(sbml_dir / "Lactobacillus_plantarum.xml"))
}

# Initial params
initial_biomass = {'OR16': 0.5, 'NS21': 0.1, 'LP': 0.1}
initial_metabolites = {
    'glc__D_e': 0.0, # Zero glucose to force rubber use
    'nh4_e': 10.0,
    'o2_e': 0.25,
    'pi_e': 10.0,
    'yeast_extract_e': 0.0 # Zero YE to force rubber use
}

sim = dFBASimulator(
    models=models,
    initial_biomass=initial_biomass,
    initial_metabolites=initial_metabolites,
    initial_rubber=100.0,
    dt=0.1
)

print("--- Testing OR16 objective fallback ---")
# Force starvation
sim.state.metabolites['glc__D_e'] = 0.0
sim.state.metabolites['yeast_extract_e'] = 0.0

# Manually call set_uptake_constraints
sim.set_uptake_constraints('OR16', sim.state.metabolites)
model = sim.models['OR16']

# Set objective to R_R_LCP manually and optimize
model.objective = 'R_R_LCP'
sol = model.optimize()
print(f"OR16 Status: {sol.status}, R_R_LCP flux: {sol.objective_value}")

# Check rubber uptake
rubber_rxn = model.reactions.get_by_id(sim.exchange_reactions['OR16']['rubber_e'])
print(f"Rubber uptake flux: {rubber_rxn.flux}")

