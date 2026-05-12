import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Rhizobacter_gummiphilus_NS21.xml"))

def check_growth(open_reactions):
    with model:
        for r in model.exchanges: r.lower_bound = 0
        for rid in open_reactions:
            if rid in model.reactions: model.reactions.get_by_id(rid).lower_bound = -1000
        return model.optimize().objective_value

all_ex = [r.id for r in model.exchanges if r.id.startswith('EX_')]
print(f"Total exchanges: {len(all_ex)}")

current_set = list(all_ex)
for rid in all_ex:
    test_set = [r for r in current_set if r != rid]
    if check_growth(test_set) > 1e-6:
        current_set.remove(rid)

print("\n--- Minimal Essential Exchange Reactions for NS21 ---")
for e in current_set:
    r = model.reactions.get_by_id(e)
    print(f"ID: {e}, Reaction: {r.reaction}")

