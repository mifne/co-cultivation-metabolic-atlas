
import cobra

def diagnose_lp():
    model = cobra.io.read_sbml_model('models/sbml/final_consortium/Lactobacillus_plantarum.xml')
    model.objective = 'R_GLYCOLIPOPROTEIN_SYN_SEC'
    for r in model.exchanges:
        r.lower_bound = -100
    
    sol = model.optimize()
    print(f"LP Max BS flux: {sol.objective_value}")
    if sol.status != 'optimal':
        print("Infeasible. Checking for gaps...")
        # Check if it can produce precursors
        rxn = model.reactions.get_by_id('R_GLYCOLIPOPROTEIN_SYN_SEC')
        for met in rxn.reactants:
            sink = model.add_boundary(met, type='sink')
            model.objective = sink
            s = model.optimize()
            print(f"  Can produce {met.id} ({met.name}): {s.objective_value}")
            model.remove_reactions([sink])

def diagnose_or16():
    model = cobra.io.read_sbml_model('models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml')
    for r in model.exchanges:
        r.lower_bound = -100
    
    pathway = ['R_LCP', 'ISOP_ALDH', 'ISOP_ACS']
    print("\n--- OR16 Pathway Flux Trace ---")
    for rid in pathway:
        if rid in model.reactions:
            rxn = model.reactions.get_by_id(rid)
            model.objective = rxn
            sol = model.optimize()
            print(f"  Max flux for {rid}: {sol.objective_value} (Status: {sol.status})")
            if sol.objective_value < 1e-6:
                print(f"    Reactants: {[(m.id, m.name) for m in rxn.reactants]}")
                print(f"    Products: {[(m.id, m.name) for m in rxn.products]}")
        else:
            print(f"  Reaction {rid} NOT FOUND")

if __name__ == "__main__":
    print("--- LP Diagnosis ---")
    diagnose_lp()
    print("\n--- OR16 Diagnosis ---")
    diagnose_or16()
