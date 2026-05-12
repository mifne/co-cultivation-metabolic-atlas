import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

def check_growth(open_reactions):
    with model:
        for r in model.exchanges: r.lower_bound = 0
        for rid in open_reactions:
            if rid in model.reactions: model.reactions.get_by_id(rid).lower_bound = -1000
        return model.optimize().objective_value

# Initial list of ALL exchanges
all_ex = [r.id for r in model.exchanges if r.id.startswith('EX_')]
print(f"Total exchanges: {len(all_ex)}")

# Binary search / refinement to find MINIMAL set
essentials = []
current_set = list(all_ex)

# Try removing in blocks
for size in [50, 10, 1]:
    new_set = []
    for i in range(0, len(current_set), size):
        subset = current_set[i:i+size]
        test_set = [r for r in current_set if r not in subset] + essentials
        if check_growth(test_set) > 1e-6:
            # We can remove this block!
            pass
        else:
            # We NEED something in this block!
            if size == 1:
                essentials.append(subset[0])
            else:
                new_set.extend(subset)
    current_set = new_set

print("\n--- Minimal Essential Exchange Reactions for OR16 ---")
for e in essentials:
    r = model.reactions.get_by_id(e)
    print(f"ID: {e}, Met: {r.reaction}")

