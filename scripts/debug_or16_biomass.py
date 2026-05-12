import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

print(f"Biomass RXN: Growth")
rxn = model.reactions.Growth
for met, coeff in rxn.metabolites.items():
    if coeff < 0:
        # Check if an exchange or source exists for this precursor
        exchanges = [r for r in met.reactions if len(r.metabolites) == 1 and r.metabolites[met] == -1]
        if not exchanges:
            print(f"Precursor: {met.id} ({met.name}) - NO EXCHANGE FOUND")
        else:
            # Check if it's open in current model
            lb = min(e.lower_bound for e in exchanges)
            print(f"Precursor: {met.id} ({met.name}) - Exchange {exchanges[0].id}, LB: {lb}")

