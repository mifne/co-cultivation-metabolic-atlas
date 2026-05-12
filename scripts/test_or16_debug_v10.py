import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

with model:
    # 1. Start from ALL open
    for r in model.exchanges: r.lower_bound = -1000
    print(f"Growth (ALL open): {model.optimize().objective_value}")
    
    # 2. Close specific ones we thought were unnecessary
    redundant = ['EX_12ppd__S_e', 'EX_14glucan_e', 'EX_2hxmp_e', 'EX_3mb_e', 'EX_acald_e', 'EX_adn_e']
    for rid in redundant: model.reactions.get_by_id(rid).lower_bound = 0
    print(f"Growth (Partial close): {model.optimize().objective_value}")

