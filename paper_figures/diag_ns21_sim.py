import cobra
import numpy as np
from pathlib import Path
from src.dfba_simulator import dFBASimulator

sbml_dir = Path("models/sbml/final_consortium")
models = {
    'NS21': cobra.io.read_sbml_model(str(sbml_dir / "Rhizobacter_gummiphilus_NS21.xml")),
}

sim = dFBASimulator(
    models=models,
    initial_biomass={'NS21': 1.0},
    initial_metabolites={'odtd_e': 10.0, 'nh4_e': 0.0, 'yeast_extract_e': 0.0},
    initial_rubber=0.0,
    dt=1.0
)

# Call step directly
state = sim.step({}, {})

ns21 = state.species['NS21']
print(f"Objective: {sim.models['NS21'].objective.expression}")
print(f"Growth Rate: {ns21.growth_rate}")
print(f"PHA accumulated: {ns21.pha_accumulated}")

