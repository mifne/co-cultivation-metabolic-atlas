import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Lactobacillus_plantarum.xml"))

# Check if it can secrete amino acids
amino_acids = ['ala__L_e', 'arg__L_e', 'asn__L_e', 'asp__L_e', 'cys__L_e', 'glu__L_e', 'gln__L_e', 'gly_e', 'his__L_e', 'ile__L_e', 'leu__L_e', 'lys__L_e', 'met__L_e', 'phe__L_e', 'pro__L_e', 'ser__L_e', 'thr__L_e', 'trp__L_e', 'tyr__L_e', 'val__L_e']

print("--- LP Secretion Potential ---")
with model:
    # Give it plenty of glucose
    model.reactions.EX_glc__D_e.lower_bound = -10
    for aa in amino_acids:
        try:
            rxn = model.reactions.get_by_id(f"EX_{aa}")
            # Maximize secretion of this AA
            model.objective = rxn
            sol = model.optimize()
            if sol.objective_value > 1e-6:
                print(f"Can secrete {aa}: {sol.objective_value}")
        except: pass
