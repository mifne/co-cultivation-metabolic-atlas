import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

def find_exchange(target_id):
    for r in model.exchanges:
        if target_id in r.reaction:
            return r.id
    return "NONE"

# Identified minimal essentials
essentials = ['ca2_e', 'cl_e', 'cobalt2_e', 'cu_e', 'k_e', 'mg2_e', 'mn2_e', 'o2_e', 'salchs4fe_e', 'tsul_e', 'ump_e', 'xtsn_e', 'zn2_e']
for m in essentials:
    print(f"Met: {m} -> Exchange: {find_exchange(m)}")

