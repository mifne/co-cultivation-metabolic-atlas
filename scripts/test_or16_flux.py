import cobra
model = cobra.io.read_sbml_model("models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml")
sol = model.optimize()
print("Growth:", sol.objective_value)
for r in model.reactions:
    if "EX_" in r.id and abs(sol.fluxes.get(r.id, 0)) > 1:
        print(f"{r.id}: {sol.fluxes.get(r.id, 0)}")
