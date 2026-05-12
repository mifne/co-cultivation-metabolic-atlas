import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Lactobacillus_plantarum.xml"))

def get_mu(subset):
    with model:
        for r in model.exchanges:
            r.bounds = (0, 1000)
        for rid in subset:
            model.reactions.get_by_id(rid).bounds = (-1000, 1000)
        if 'EX_glc__D_e' in model.reactions:
            model.reactions.EX_glc__D_e.lower_bound = -10
        return model.optimize().objective_value

all_ex = [r.id for r in model.exchanges]
current_set = list(all_ex)
print(f"Initial LP MU: {get_mu(current_set)}")

essentials = []
for rid in all_ex:
    test_set = [r for r in current_set if r != rid]
    if get_mu(test_set) > 1e-6:
        current_set.remove(rid)
    else:
        essentials.append(rid)

print("\n--- Absolute Minimal Essential Reactions for LP ---")
for e in essentials:
    r = model.reactions.get_by_id(e)
    print(f"ID: {e}, Reaction: {r.reaction}")
