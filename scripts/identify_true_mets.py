import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

def find_best_match(target):
    # Try exact match
    if target in model.metabolites: return target
    # Try with M_ prefix
    if f"M_{target}" in model.metabolites: return f"M_{target}"
    # Try case insensitive or partial
    matches = [m.id for m in model.metabolites if target.lower() in m.id.lower()]
    return matches[:5] # Return top 5

targets = ['accoa_c', 'ppcoa_c', 'atp_c', 'adp_c', 'nad_c', 'nadh_c', 'fad_c', 'fadh2_c', 'coa_c', 'pi_c', 'ppi_c', 'amp_c', 'h2o_c', 'h_c']
for t in targets:
    print(f"Target: {t} -> Match: {find_best_match(t)}")

# Also check NS21
print("\n--- NS21 ---")
model2 = cobra.io.read_sbml_model(str(sbml_dir / "Rhizobacter_gummiphilus_NS21.xml"))
for t in targets:
    # Check exact in model2
    match = t if t in model2.metabolites else (f"M_{t}" if f"M_{t}" in model2.metabolites else "NONE")
    print(f"Target: {t} -> Match: {match}")

