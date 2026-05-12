import numpy as np
import pandas as pd
from pathlib import Path
from main import load_sbml_models, select_consortium_models, get_initial_params
from src.dfba_simulator import dFBASimulator

def run_flux_test(scenario: str):
    sbml_dir = Path('models/sbml/final_consortium')
    all_models = load_sbml_models(sbml_dir)
    
    if scenario == 'OR16_only':
        models = {k: v for k, v in all_models.items() if 'OR16' in k}
    elif scenario == 'NS21_only':
        models = {k: v for k, v in all_models.items() if 'NS21' in k}
    else:
        models = select_consortium_models(all_models)
        
    initial_biomass, initial_metabolites = get_initial_params(models)
    
    # 必須無機塩類のみを残し、すべての有機炭素源を 0 にする
    for met in initial_metabolites.keys():
        if met not in ['nh4_e', 'o2_e', 'pi_e', 'so4_e', 'ca2_e', 'mg2_e', 'fe3_e', 'fe2_e', 'mn2_e', 'zn2_e', 'cu2_e', 'cobalt2_e', 'cl_e', 'k_e', 'na_e', 'ni2_e', 'h_e', 'h2o_e', 'co2_e', 'mobd_e']:
            initial_metabolites[met] = 0.0

    initial_metabolites['nh4_e'] = 100.0
    initial_metabolites['o2_e'] = 100.0
    for met in ['pi_e', 'so4_e', 'ca2_e', 'mg2_e', 'fe3_e', 'mn2_e', 'zn2_e', 'cu2_e', 'cobalt2_e', 'cl_e', 'k_e', 'na_e', 'ni2_e']:
        initial_metabolites[met] = 100.0


    
    sim = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=100.0,
        volume=1.0,
        dt=1.0,
        carrying_capacity=20.0
    )
    
    print(f"\n--- Testing Scenario: {scenario} ---")
    
    # Run 1 step to see initial fluxes
    sim.step({}, {})
    
    for name, state in sim.state.species.items():
        print(f"[{name}] Biomass: {state.biomass:.4f}, Growth Rate: {state.growth_rate:.4f}")
        
        # Check specific fluxes related to rubber and C30/ODTD
        print("  Key Fluxes (Secretion > 0, Uptake < 0):")
        fluxes = {}
        for met_id in state.metabolite_secretion.keys():
            if state.metabolite_secretion[met_id] > 0:
                print(f"    {met_id}: {state.metabolite_secretion[met_id]:.4f} (Secretion)")
        for met_id in state.metabolite_uptake.keys():
            if state.metabolite_uptake[met_id] > 0:
                print(f"    {met_id}: {-state.metabolite_uptake[met_id]:.4f} (Uptake)")
    
    print(f"Environmental Pool -> Rubber: {sim.state.rubber_concentration:.4f}, C30: {sim.state.metabolites.get('C30_oligo_e', 0.0):.4f}, ODTD: {sim.state.metabolites.get('odtd_e', 0.0):.4f}")

if __name__ == '__main__':
    run_flux_test('OR16_only')
    run_flux_test('NS21_only')
    run_flux_test('Consortium')
