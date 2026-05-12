import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

# Identify actual current bounds
print("--- Initial Exchange Bounds ---")
for r in model.exchanges:
    if r.lower_bound < 0:
        print(f"Reaction {r.id}: {r.lower_bound} to {r.upper_bound} ({r.reaction})")

print("\n--- Testing default growth ---")
sol = model.optimize()
print(f"Default growth: {sol.objective_value}")

