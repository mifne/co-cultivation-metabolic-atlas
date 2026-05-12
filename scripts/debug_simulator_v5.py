import cobra
import numpy as np
from pathlib import Path
from src.dfba_simulator import dFBASimulator

sbml_dir = Path("models/sbml/final_consortium")
models = {
    'OR16': cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml")),
}

sim = dFBASimulator(
    models=models,
    initial_biomass={'OR16': 1.0},
    initial_metabolites={'glc__D_e': 0.0, 'yeast_extract_e': 1.0}, # Force YE use
    initial_rubber=100.0
)

m = sim.models['OR16']
sim.set_uptake_constraints('OR16', sim.state.metabolites)

print("--- OR16 Constraints with YE (Full Debug) ---")
for r in m.exchanges:
    if r.lower_bound < 0:
        print(f"Reaction {r.id}: LB={r.lower_bound}, RB={r.reaction}")

sol = m.optimize()
print(f"\nGrowth with YE: {sol.objective_value}")
if sol.objective_value < 1e-6:
    print("\n--- Identifying why it fails ---")
    from cobra.flux_analysis import pfba
    try:
        # Find which YE components were missed or NOT open
        ye_list = sim.YE_COMPONENTS
        for c in ye_list:
            rid = sim.exchange_reactions['OR16'].get(c)
            if rid:
                lb = m.reactions.get_by_id(rid).lower_bound
                if lb >= 0: print(f"ALERT: YE component {c} ({rid}) IS CLOSED!")
            else:
                print(f"ALERT: YE component {c} HAS NO REACTION IN OR16!")
    except: pass

