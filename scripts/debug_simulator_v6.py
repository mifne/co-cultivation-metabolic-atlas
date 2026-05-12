import cobra
import numpy as np
from pathlib import Path
from src.dfba_simulator import dFBASimulator

sbml_dir = Path("models/sbml/final_consortium")
models = {
    'OR16': cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml")),
}

# Starvation case
sim = dFBASimulator(
    models=models,
    initial_biomass={'OR16': 1.0},
    initial_metabolites={'glc__D_e': 0.0, 'yeast_extract_e': 0.0}, 
    initial_rubber=100.0
)

m = sim.models['OR16']
sim.set_uptake_constraints('OR16', sim.state.metabolites)

print("--- OR16 Constraints in Starvation ---")
for r in m.exchanges:
    if r.lower_bound < 0:
        print(f"Reaction {r.id}: LB={r.lower_bound}, RB={r.reaction}")

# Check R_LCP
print(f"\nR_LCP bound: {m.reactions.R_LCP.upper_bound}")

sol = m.optimize()
print(f"Growth: {sol.objective_value}")

# Switch objective to R_LCP
m.objective = 'R_LCP'
sol2 = m.optimize()
print(f"R_LCP Objective value: {sol2.objective_value}")

