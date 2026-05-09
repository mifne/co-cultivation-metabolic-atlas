import numpy as np
from pathlib import Path
from main import load_sbml_models, select_consortium_models, get_initial_params
from src.dfba_simulator import dFBASimulator

def reproduce_crash():
    print("Loading models...")
    sbml_dir = Path('models/sbml/final_consortium')
    all_models = load_sbml_models(sbml_dir)
    models = select_consortium_models(all_models)
    
    initial_biomass, initial_metabolites = get_initial_params(models)
    
    test_metabolites = initial_metabolites.copy()
    test_metabolites['o2_e'] = 0.25 # 飽和
    test_metabolites['glc__D_e'] = 0.0 # 枯渇
    test_metabolites['nh4_e'] = 0.0 # 枯渇
    test_metabolites['rubber_fragment_e'] = 100.0 # 豊富
    test_metabolites['odtd_e'] = 100.0 # 豊富

    print("Initializing simulator...")
    sim = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=test_metabolites,
        initial_rubber=100.0,
        volume=1.0,
        dt=0.2
    )

    print("Running solve_fba for each species with various bounds...")
    for species_name in models.keys():
        print(f"\n--- Testing {species_name} ---")
        
        sim.set_uptake_constraints(species_name, test_metabolites, dynamic_kla=200.0)
        
        model = sim.models[species_name]
        try:
            model.solver.configuration.presolve = True
            sol = model.optimize()
            print(f"  Presolve=True:  Status = {sol.status}, Obj = {sol.objective_value}")
        except Exception as e:
            print(f"  Presolve=True:  CRASH -> {e}")
            
        try:
            model.solver.configuration.presolve = False
            sol = model.optimize()
            print(f"  Presolve=False: Status = {sol.status}, Obj = {sol.objective_value}")
        except Exception as e:
            print(f"  Presolve=False: CRASH -> {e}")
            
        print(f"  [Key Bounds]")
        for met_id in ['rubber_e', 'pha_c', 'o2_e', 'glc__D_e', 'nh4_e', 'odtd_e']:
            rxn_id = sim.exchange_reactions[species_name].get(met_id)
            if rxn_id and rxn_id in model.reactions:
                rxn = model.reactions.get_by_id(rxn_id)
                print(f"    {rxn_id}: LB={rxn.lower_bound:.4f}, UB={rxn.upper_bound:.4f}")

if __name__ == "__main__":
    reproduce_crash()
