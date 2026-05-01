import cobra

model_path = "models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml"
model = cobra.io.read_sbml_model(model_path)

print("Total reactions:", len(model.reactions))
print("Searching for reactions containing 'LATA' or 'rubber' or 'PHA':")
for rxn in model.reactions:
    if any(x in rxn.id.upper() for x in ["LATA", "RUBBER", "PHA"]):
        print(f"ID: {rxn.id}, Name: {rxn.name}")

print("\nSearching for metabolites containing 'rubber' or 'PHA' or 'ODTD':")
for met in model.metabolites:
    if any(x in met.id.upper() for x in ["RUBBER", "PHA", "ODTD"]):
        print(f"ID: {met.id}, Name: {met.name}")
