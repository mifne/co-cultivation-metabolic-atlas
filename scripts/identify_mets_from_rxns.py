import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

rxns = ['EX_ca2_e', 'EX_cl_e', 'EX_cobalt2_e', 'EX_cu_e', 'EX_k_e', 'EX_mg2_e', 'EX_mn2_e', 'EX_o2_e', 'EX_salchs4fe_e', 'EX_tsul_e', 'EX_ump_e', 'EX_xtsn_e', 'EX_zn2_e']
for rid in rxns:
    r = model.reactions.get_by_id(rid)
    met = list(r.metabolites.keys())[0]
    mid = met.id
    while mid.startswith('M_'): mid = mid[2:]
    print(f"Reaction: {rid} -> Met ID: {mid}")
