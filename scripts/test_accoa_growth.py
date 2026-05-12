import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

with model:
    # 1. Close all carbon
    for r in model.exchanges:
        if r.lower_bound < 0: r.lower_bound = 0
    # 2. Open standard medium
    for m in ['o2_e', 'nh4_e', 'pi_e', 'so4_e', 'mg2_e', 'ca2_e', 'k_e', 'fe2_e', 'fe3_e', 'h_e', 'h2o_e']:
        try: model.reactions.get_by_id(f"R_EX_{m[2:]}").lower_bound = -1000
        except:
            try: model.reactions.get_by_id(f"EX_{m[2:]}").lower_bound = -1000
            except: pass
    
    # 3. Add a source of accoa_c
    try:
        accoa = model.metabolites.accoa_c
        source = model.add_boundary(accoa, type='sink', lb=-10, ub=0) # sink as a source
        source.id = 'TEST_ACCOA_SOURCE'
        
        sol = model.optimize()
        print(f"Growth on internal Acetyl-CoA source: {sol.objective_value}")
    except Exception as e:
        print(f"Error adding source: {e}")

    # 4. Check if it needs Propionyl-CoA too
    try:
        ppcoa = model.metabolites.ppcoa_c
        source2 = model.add_boundary(ppcoa, type='sink', lb=-10, ub=0)
        source2.id = 'TEST_PPCOA_SOURCE'
        sol2 = model.optimize()
        print(f"Growth on internal ACCOA + PPCOA sources: {sol2.objective_value}")
    except Exception as e:
        print(f"Error adding source2: {e}")

