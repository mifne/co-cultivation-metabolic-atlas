import numpy as np
from pathlib import Path
from main import load_sbml_models, select_consortium_models, get_initial_params
from src.dfba_simulator import dFBASimulator

def reproduce_crash():
    sbml_dir = Path('models/sbml/final_consortium')
    all_models = load_sbml_models(sbml_dir)
    models = select_consortium_models(all_models)
    initial_biomass, initial_metabolites = get_initial_params(models)
    
    sim = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=100.0,
        dt=0.2
    )
    
    # クラッシュ直前のログを模倣
    for species_name, model in models.items():
        print(f"\n[{species_name}] Presolve config check:")
        sim.set_uptake_constraints(species_name, initial_metabolites, dynamic_kla=0.0) # kLa 0
        model.solver.configuration.presolve = True
        try:
            sol = model.optimize()
            print(f"  kLa=0, Presolve=True: {sol.status}")
        except Exception as e:
            print(f"  kLa=0, Presolve=True: CRASH -> {e}")

if __name__ == "__main__":
    reproduce_crash()
