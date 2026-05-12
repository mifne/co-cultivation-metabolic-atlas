import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

def check_growth(open_reactions):
    with model:
        for r in model.exchanges: r.lower_bound = 0
        for rid in open_reactions:
            if rid in model.reactions: model.reactions.get_by_id(rid).lower_bound = -1000
        return model.optimize().objective_value

# Initial minimal set + carbon source
essentials = ['EX_ca2_e', 'EX_cl_e', 'EX_cobalt2_e', 'EX_cu_e', 'EX_k_e', 'EX_mg2_e', 'EX_mn2_e', 'EX_o2_e', 'EX_salchs4fe_e', 'EX_tsul_e', 'EX_ump_e', 'EX_xtsn_e', 'EX_zn2_e', 'EX_glc__D_e']

print(f"Growth with refined essentials + Glucose: {check_growth(essentials)}")

# What about Nitrogen and Phosphorus?
# In the search, they might have been part of UMP or something else?
# Let's check which ones provide N and P
for rid in essentials:
    r = model.reactions.get_by_id(rid)
    print(f"ID: {rid}, Reaction: {r.reaction}")

