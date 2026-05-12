import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

all_ex = [r.id for r in model.exchanges]
current_set = list(all_ex)

def get_mu(subset):
    with model:
        for r in model.exchanges: r.lower_bound = 0
        for rid in subset: model.reactions.get_by_id(rid).lower_bound = -1000
        return model.optimize().objective_value

# Initial check
print(f"Initial MU: {get_mu(current_set)}")

# Remove one by one
essentials = []
for rid in all_ex:
    test_set = [r for r in current_set if r != rid]
    if get_mu(test_set) > 1e-6:
        current_set.remove(rid)
    else:
        essentials.append(rid)

print("\n--- Absolute Minimal Essential Reactions for OR16 ---")
for e in essentials:
    r = model.reactions.get_by_id(e)
    print(f"ID: {e}, Reaction: {r.reaction}")

