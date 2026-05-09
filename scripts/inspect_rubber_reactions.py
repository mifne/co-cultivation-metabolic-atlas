from pathlib import Path
from main import load_sbml_models, select_consortium_models

def inspect_rubber_logic():
    models = select_consortium_models(load_sbml_models(Path('models/sbml/final_consortium')))
    for name, model in models.items():
        print(f"\n[{name}]")
        target_rxns = [
            'EX_rubber_bulk_e', 'EX_rubber_fragment_e', 'EX_odtd_e', 'EX_rubber_e',
            'R_LCP', 'R_ROX', 'LCP', 'ROX'
        ]
        for rid in target_rxns:
            if rid in model.reactions:
                rxn = model.reactions.get_by_id(rid)
                print(f"  {rid}: {rxn.reaction} ({rxn.lower_bound}, {rxn.upper_bound})")
        
        # Check metabolites involved in rubber degradation
        for met in model.metabolites:
            if 'rubber' in met.id or 'odtd' in met.id:
                print(f"  Metabolite: {met.id} ({met.name})")

if __name__ == "__main__":
    inspect_rubber_logic()
