import cobra
model = cobra.io.read_sbml_model("models/sbml/final_consortium/Lactobacillus_plantarum.xml")

# Try different reaction IDs for glucose exchange
glc_rxn = None
for r in model.reactions:
    if "glc" in r.id.lower() and "EX" in r.id:
        glc_rxn = r
        break

if glc_rxn:
    glc_rxn.lower_bound = -10
    sol = model.optimize()
    print("Growth:", sol.objective_value)
    print(f"{glc_rxn.id} flux:", sol.fluxes.get(glc_rxn.id))
    
    lac_flux = 0
    for r in model.reactions:
        if "lac" in r.id.lower() and "EX" in r.id:
            lac_flux += sol.fluxes.get(r.id, 0)
    print("Lac flux:", lac_flux)
    
    h_flux = 0
    for r in model.reactions:
        if "h_e" in r.id.lower() and "EX" in r.id:
            h_flux += sol.fluxes.get(r.id, 0)
    print("H+ flux:", h_flux)
else:
    print("Glucose exchange reaction not found.")