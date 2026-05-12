import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))
for rxn in model.reactions:
    if 'C30' in rxn.id or 'LCP' in rxn.id or 'rubber' in rxn.id:
        print(f"ID: {rxn.id}, Name: {rxn.name}")
