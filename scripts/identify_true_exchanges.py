import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

def find_exchange(target_id):
    # Try exact match first
    for r in model.exchanges:
        if target_id in r.reaction:
            return r.id
    return "NONE"

standard_mets = ['glc__D_e', 'nh4_e', 'pi_e', 'so4_e', 'mg2_e', 'ca2_e', 'k_e', 'fe2_e', 'fe3_e', 'h_e', 'h2o_e', 'o2_e']
for m in standard_mets:
    print(f"Met: {m} -> Exchange: {find_exchange(m)}")

