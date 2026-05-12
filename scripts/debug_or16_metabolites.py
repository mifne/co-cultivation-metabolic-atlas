import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

# Check rubber related species
for met in model.metabolites:
    if 'rubber' in met.id or 'C30' in met.id:
        print(f"Metabolite: {met.id} ({met.name})")

# Check R_LCP
print("\n--- R_LCP ---")
rxn = model.reactions.get_by_id("R_LCP")
print(f"Reaction: {rxn.id}")
print(f"Formula: {rxn.reaction}")

# Check R_C30t
print("\n--- R_C30t ---")
rxn = model.reactions.get_by_id("R_C30t")
print(f"Reaction: {rxn.id}")
print(f"Formula: {rxn.reaction}")

# Check R_C30_cat
print("\n--- R_C30_cat ---")
rxn = model.reactions.get_by_id("R_C30_cat")
print(f"Reaction: {rxn.id}")
print(f"Formula: {rxn.reaction}")

# Check if Acetyl-CoA and Propionyl-CoA exist
print("\n--- Central Metabolites ---")
for mid in ['accoa_c', 'ppcoa_c', 'atp_c', 'nad_c', 'fad_c', 'coa_c']:
    found = False
    for m in model.metabolites:
        if mid in m.id:
            print(f"Found {mid}: {m.id}")
            found = True
            break
    if not found: print(f"NOT FOUND: {mid}")
