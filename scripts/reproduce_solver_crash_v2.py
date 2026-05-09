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

    # 実際のstep()ループを模倣
    for t in range(50):
        print(f"Step {t} (Time {sim.state.time:.2f})")
        # RLアクションの模倣 (ランダムまたは極端な値)
        nutrient_supp = {
            'sn_or16': np.random.uniform(0, 0.1),
            'sn_ns21': np.random.uniform(0, 0.1),
            'sn_lp': np.random.uniform(0, 0.1),
            'yeast_extract': np.random.uniform(0, 0.5)
        }
        dynamic_kla = np.random.uniform(0, 200)
        try:
            sim.step({}, nutrient_supp, dynamic_kla=dynamic_kla)
        except Exception as e:
            print(f"CRASH AT STEP {t}: {e}")
            break

if __name__ == "__main__":
    reproduce_crash()
