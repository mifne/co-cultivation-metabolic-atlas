import cobra
from pathlib import Path
from src.dfba_simulator import dFBASimulator

sbml_dir = Path('models/sbml/final_consortium')
models = {
    'NS21': cobra.io.read_sbml_model(str(sbml_dir / 'Rhizobacter_gummiphilus_NS21.xml')),
}

sim = dFBASimulator(
    models=models,
    initial_biomass={'NS21': 0.1},
    initial_metabolites={},
    dt=0.2
)

key = list(sim.state.species.keys())[0]
print(f'Exchange reactions mapping for pha_c: {sim.exchange_reactions[key].get("pha_c")}')

# Also list all pha reactions in the model
print("All reactions with 'pha' in ID:")
for r in models['NS21'].reactions:
    if 'pha' in r.id.lower():
        print(f"{r.id}: {r.name}")

