import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Rhizobacter_gummiphilus_NS21.xml"))

rxn = model.reactions.get_by_id("EX_h_e")
print(f"H+ exchange bounds: {rxn.lower_bound} to {rxn.upper_bound}")

# Find reactions producing/consuming h_c
h_c = model.metabolites.h_c
print(f"\n--- Top 10 reactions involving h_c ---")
for r in sorted(h_c.reactions, key=lambda x: abs(x.metabolites[h_c]), reverse=True)[:10]:
    print(f"{r.id}: {r.reaction}")

