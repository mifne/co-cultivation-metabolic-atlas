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

sim = dFBASimulator(
    models=models,
    initial_biomass={'OR16': 0.5, 'NS21': 0.1, 'LP': 0.1},
    initial_metabolites={'glc__D_e': 1.0},
    initial_rubber=100.0
)

print("--- OR16 Exchange Map ---")
for met, rid in sim.exchange_reactions['OR16'].items():
    if 'rubber' in met or 'C30' in met or 'odtd' in met:
        print(f"Met: {met} -> Reaction: {rid}")

print("\n--- NS21 Exchange Map ---")
for met, rid in sim.exchange_reactions['NS21'].items():
    if 'rubber' in met or 'C30' in met or 'odtd' in met:
        print(f"Met: {met} -> Reaction: {rid}")

print("\n--- Testing Degradation Trigger ---")
# Force rubber uptake in OR16
# We need to simulate a solution where EX_rubber_bulk_e has flux
sim.state.species['OR16'].metabolite_uptake['rubber_e'] = 1.0
sim.state.metabolites['biosurfactant_e'] = 0.0
sim.degrade_rubber({})
print(f"Rubber after dummy uptake: {sim.state.rubber_concentration}")

