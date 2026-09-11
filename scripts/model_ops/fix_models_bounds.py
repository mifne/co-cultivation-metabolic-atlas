import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import cobra
import os

def fix_mandatory_uptake():
    sbml_dir = 'models/sbml/final_consortium'
    for f in os.listdir(sbml_dir):
        if f.endswith('.xml'):
            path = os.path.join(sbml_dir, f)
            model = cobra.io.read_sbml_model(path)
            changed = False
            for rxn in model.exchanges:
                if rxn.upper_bound < 0:
                    print(f"Fixing {f}: {rxn.id} upper_bound {rxn.upper_bound} -> 0")
                    rxn.upper_bound = 0
                    changed = True
            if changed:
                cobra.io.write_sbml_model(model, path)

if __name__ == "__main__":
    fix_mandatory_uptake()
