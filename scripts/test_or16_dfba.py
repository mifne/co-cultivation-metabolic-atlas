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

# Initial params - force starvation
initial_biomass = {'OR16': 0.5, 'NS21': 0.1, 'LP': 0.1}
initial_metabolites = {
    'glc__D_e': 0.0,
    'nh4_e': 100.0,
    'o2_e': 0.25,
    'pi_e': 100.0,
    'yeast_extract_e': 0.0,
    'h2o_e': 55000.0,
    'h_e': 1e-4
}

sim = dFBASimulator(
    models=models,
    initial_biomass=initial_biomass,
    initial_metabolites=initial_metabolites,
    initial_rubber=100.0,
    dt=0.1
)

# Run one step
print("--- Running one step in starvation ---")
state = sim.step({}, {})

print(f"Time: {state.time}")
print(f"OR16 Biomass: {state.species['OR16'].biomass}")
print(f"OR16 Growth Rate: {state.species['OR16'].growth_rate}")
print(f"Rubber Concentration: {state.rubber_concentration}")

# Check OR16 uptake
uptake = state.species['OR16'].metabolite_uptake
print(f"OR16 Uptake: {uptake}")

# Check R_LCP flux specifically
model = sim.models['OR16']
sol = model.optimize()
print(f"OR16 Sol Status: {sol.status}, Obj: {sol.objective_value}")
if 'R_LCP' in sol.fluxes:
    print(f"R_LCP flux: {sol.fluxes['R_LCP']}")

