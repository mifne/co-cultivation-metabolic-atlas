import cobra
from main import load_sbml_models, select_consortium_models, get_initial_params
from pathlib import Path

sbml_dir = Path('models/sbml/final_consortium')
all_models = load_sbml_models(sbml_dir)
models = select_consortium_models(all_models)

for name, model in models.items():
    print(f"\n--- Testing NH4+ uptake in {name} ---")
    if "OR16" in name:
        model.reactions.get_by_id("EX_nh4_e").lower_bound = -10
    elif "NS21" in name:
        model.reactions.get_by_id("EX_nh4_e").lower_bound = -10
    else:
        # LP
        pass 

    try:
        sol = model.optimize()
        for r in model.reactions:
            if "EX_nh4_e" in r.id or "R_EX_nh4_e" in r.id:
                nh4_flux = sol.fluxes.get(r.id, 0)
                print(f"NH4+ flux ({r.id}): {nh4_flux}")
            if "EX_h_e" in r.id or "R_EX_h_e" in r.id:
                h_flux = sol.fluxes.get(r.id, 0)
                print(f"H+ flux ({r.id}): {h_flux}")
    except Exception as e:
        print(f"Error: {e}")
