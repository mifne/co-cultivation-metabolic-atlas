import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Rhizobacter_gummiphilus_NS21.xml"))

all_ex = [r.id for r in model.exchanges]
current_set = list(all_ex)

def get_mu(subset):
    with model:
        for r in model.exchanges: r.lower_bound = 0
        for rid in subset: model.reactions.get_by_id(rid).lower_bound = -1000
        # Add carbon
        model.reactions.EX_glc__D_e.lower_bound = -10
        return model.optimize().objective_value

print(f"Initial NS21 MU: {get_mu(current_set)}")

essentials = []
for rid in all_ex:
    test_set = [r for r in current_set if r != rid]
    if get_mu(test_set) > 1e-6:
        current_set.remove(rid)
    else:
        essentials.append(rid)

print("\n--- Absolute Minimal Essential Reactions for NS21 ---")
for e in essentials:
    r = model.reactions.get_by_id(e)
    print(f"ID: {e}, Reaction: {r.reaction}")
