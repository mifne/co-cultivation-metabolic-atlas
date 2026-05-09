import numpy as np
from pathlib import Path
from main import load_sbml_models, select_consortium_models, get_initial_params
from src.dfba_simulator import dFBASimulator

def diagnose_metabolic_limits():
    sbml_dir = Path('models/sbml/final_consortium')
    all_models = load_sbml_models(sbml_dir)
    models = select_consortium_models(all_models)
    initial_biomass, initial_metabolites = get_initial_params(models)
    
    # Setup sim
    sim = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=100.0,
        volume=1.0,
        dt=0.2
    )
    
    # 栄養豊富な状態にしてみる
    test_metabolites = sim.state.metabolites.copy()
    test_metabolites['o2_e'] = 0.25 # 飽和濃度
    test_metabolites['glc__D_e'] = 10.0
    test_metabolites['nh4_e'] = 10.0
    
    print("--- Diagnostic: Metabolism Limits ---")
    for species_name in models.keys():
        print(f"\n[Species: {species_name}]")
        sim.set_uptake_constraints(species_name, test_metabolites)
        model = sim.models[species_name]
        
        for met in ['o2_e', 'glc__D_e', 'nh4_e', 'rubber_fragment_e', 'odtd_e', 'rubber_e']:
            if met in sim.exchange_reactions[species_name]:
                rxn_id = sim.exchange_reactions[species_name][met]
                rxn = model.reactions.get_by_id(rxn_id)
                print(f"  {met} ({rxn_id}): LB={rxn.lower_bound:.6f}, UB={rxn.upper_bound:.6f}")
                
                # max_physical_flux の計算を再現
                conc = test_metabolites.get(met, 0.0)
                current_biomass = max(1e-6, sim.state.species[species_name].biomass)
                max_physical_flux = conc * sim.volume / (current_biomass * sim.dt)
                print(f"    (Calculated max_physical_flux: {max_physical_flux:.6f})")

if __name__ == "__main__":
    diagnose_metabolic_limits()
