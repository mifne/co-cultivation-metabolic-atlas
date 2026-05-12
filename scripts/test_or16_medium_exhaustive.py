import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

# 1. Close ALL
for r in model.exchanges:
    r.lower_bound = 0

# 2. Open standard + glucose
for m in ['EX_nh4_e', 'EX_g3pi_e', 'EX_aso4_e', 'EX_mg2_e', 'EX_ca2_e', 'EX_k_e', 'EX_fe2_e', 'EX_fe3_e', 'EX_h_e', 'EX_h2o_e', 'EX_co2_e', 'EX_glc__D_e']:
    if m in model.reactions: model.reactions.get_by_id(m).lower_bound = -1000

# 3. Add reactions from default medium until growth starts
potential_essentials = [
    'EX_ala__L_e', 'EX_arg__L_e', 'EX_asn__L_e', 'EX_asp__L_e', 'EX_cys__L_e', 'EX_glu__L_e', 'EX_gln__L_e', 
    'EX_gly_e', 'EX_his__L_e', 'EX_ile__L_e', 'EX_leu__L_e', 'EX_lys__L_e', 'EX_met__L_e', 'EX_phe__L_e', 
    'EX_pro__L_e', 'EX_ser__L_e', 'EX_thr__L_e', 'EX_trp__L_e', 'EX_tyr__L_e', 'EX_val__L_e',
    'EX_ade_e', 'EX_gua_e', 'EX_ura_e', 'EX_cytd_e', 'EX_thm_e', 'EX_btn_e', 'EX_nac_e', 'EX_ribflv_e'
]

with model:
    for rid in potential_essentials:
        if rid in model.reactions: model.reactions.get_by_id(rid).lower_bound = -1000
    
    # Try all default exchanges to be absolutely sure
    default_exchanges = [r.id for r in model.exchanges if r.id.startswith('EX_')]
    for rid in default_exchanges:
        model.reactions.get_by_id(rid).lower_bound = -1000
        
    sol = model.optimize()
    print(f"Growth with ALL possible exchanges open: {sol.objective_value}")

