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

sim = dFBASimulator(
    models=models,
    initial_biomass={'OR16': 1.0, 'NS21': 1.0, 'LP': 1.0},
    initial_metabolites={'glc__D_e': 10.0, 'nh4_e': 10.0, 'pi_e': 10.0},
    initial_rubber=100.0
)

print(f"YE_COMPONENTS: {sim.YE_COMPONENTS}")
print("\n--- OR16 Exchange Mapping ---")
for mid in sim.YE_COMPONENTS:
    rid = sim.exchange_reactions['OR16'].get(mid, "NONE")
    print(f"Met: {mid} -> Reaction: {rid}")

print("\n--- OR16 Final Check before Step ---")
# This is what set_uptake_constraints does
sim.set_uptake_constraints('OR16', sim.state.metabolites)
m = sim.models['OR16']
sol = m.optimize()
print(f"OR16 Growth on Glucose (within simulator context): {sol.objective_value}")

