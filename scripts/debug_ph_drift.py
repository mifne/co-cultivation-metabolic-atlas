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

# Initial metabolites with 1.0 YE
initial_metabolites = {
    'glc__D_e': 0.0, 'nh4_e': 10.0, 'pi_e': 10.0, 'yeast_extract_e': 1.0,
    'o2_e': 0.25, 'h2o_e': 55000.0, 'h_e': 1e-4
}

sim = dFBASimulator(
    models=models,
    initial_biomass={'OR16': 0.5, 'NS21': 0.1, 'LP': 0.1},
    initial_metabolites=initial_metabolites,
    dt=0.2
)

print("--- Step 1 Analysis ---")
# Perform one step manually to see fluxes
# Instead of step(), let's just do FBA
for name in models:
    sim.set_uptake_constraints(name, sim.state.metabolites)
    sol = sim.models[name].optimize()
    if sol.status == 'optimal':
        # Find H+ flux
        h_met = 'h_e'
        rid = sim.exchange_reactions[name].get(h_met)
        if rid:
            h_flux = sol.fluxes[rid]
            print(f"{name} H+ flux (EX_{rid}): {h_flux}")
            # If negative, it means UPTAKE (raising pH)
            if h_flux < 0:
                print(f"  {name} is CONSUMING H+ (raising pH)")
                # Find why. Check other active fluxes
                print(f"  Top uptake fluxes for {name}:")
                for r, f in sol.fluxes.items():
                    if f < -1: print(f"    {r}: {f}")
                print(f"  Top secretion fluxes for {name}:")
                for r, f in sol.fluxes.items():
                    if f > 1: print(f"    {r}: {f}")
