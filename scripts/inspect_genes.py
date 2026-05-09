from pathlib import Path
from main import load_sbml_models, select_consortium_models

def inspect_genes():
    models = select_consortium_models(load_sbml_models(Path('models/sbml/final_consortium')))
    for name, model in models.items():
        print(f"\n[{name}]")
        target_rxns = ['R_LCP', 'R_ROXA', 'R_ROXB', 'R_ODTD_cat', 'EX_pha_c', 'EX_phb_c']
        for rid in target_rxns:
            if rid in model.reactions:
                rxn = model.reactions.get_by_id(rid)
                print(f"  Reaction ID: {rid}")
                print(f"    Reaction: {rxn.reaction}")
                print(f"    GPR: {rxn.gene_reaction_rule}")

if __name__ == "__main__":
    inspect_genes()
