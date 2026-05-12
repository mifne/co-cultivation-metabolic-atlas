import cobra
import numpy as np
from pathlib import Path
from src.dfba_simulator import dFBASimulator

sbml_dir = Path("models/sbml/final_consortium")
models = {
    'OR16': cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml")),
    'NS21': cobra.io.read_sbml_model(str(sbml_dir / "Rhizobacter_gummiphilus_NS21.xml")),
    'LP': cobra.io.read_sbml_model(str(sbml_dir / "Lactobacillus_plantarum.xml"))
}

# Initial metabolites with 1.0 YE (which gives 1.0 glucose)
initial_metabolites = {
    'glc__D_e': 0.0,
    'nh4_e': 10.0,
    'pi_e': 10.0,
    'yeast_extract_e': 1.0
}

sim = dFBASimulator(
    models=models,
    initial_biomass={'OR16': 0.5, 'NS21': 0.1, 'LP': 0.1},
    initial_metabolites=initial_metabolites,
    initial_rubber=100.0,
    dt=0.2
)

print(f"Initial Rubber: {sim.state.rubber_concentration}")

# Run 100 steps
for i in range(100):
    sim.step({}, {})

print(f"Final Rubber after 100 steps (20 hours): {sim.state.rubber_concentration}")
print(f"C30 oligo: {sim.state.metabolites.get('C30_oligo_e')}")
print(f"ODTD: {sim.state.metabolites.get('odtd_e')}")
print(f"pH: {-np.log10(sim.state.metabolites.get('h_e')/1000.0)}")

