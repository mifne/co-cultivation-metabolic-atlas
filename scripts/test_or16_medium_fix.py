import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

# 1. Close ALL carbon and non-essential nutrient exchanges
for r in model.exchanges:
    r.lower_bound = 0

# 2. Open MINIMAL essentials (inorganic + O2)
# Check exact reaction IDs from identification script
essentials = ['EX_nh4_e', 'EX_g3pi_e', 'EX_aso4_e', 'EX_mg2_e', 'EX_ca2_e', 'EX_k_e', 'EX_fe2_e', 'EX_fe3_e', 'EX_h_e', 'EX_h2o_e', 'EX_co2_e', 'EX_zn2_e', 'EX_cl_e']
for rid in essentials:
    if rid in model.reactions:
        model.reactions.get_by_id(rid).lower_bound = -1000

# 3. Check growth on Glucose
with model:
    model.reactions.EX_glc__D_e.lower_bound = -10
    sol_glc = model.optimize()
    print(f"Growth on Glucose (minimal medium): {sol_glc.objective_value}")

# 4. Check growth on Rubber
with model:
    model.reactions.EX_rubber_bulk_e.lower_bound = -10
    sol_rub = model.optimize()
    print(f"Growth on Rubber (minimal medium): {sol_rub.objective_value}")

# 5. If both 0, check what default medium had that we don't
if sol_glc.objective_value < 1e-6:
    print("\n--- Identifying missing essential nutrients ---")
    # Restore default and then close one by one? No, too slow.
    # Let's find reactions that ARE open in default but not in minimal.
    default_open = ['EX_12ppd__S_e', 'EX_14glucan_e', 'EX_23camp_e', 'EX_23ccmp_e', 'EX_23cgmp_e', 'EX_23cump_e', 'EX_23dhbzs3_e', 'EX_2hxmp_e', 'EX_3amp_e', 'EX_3cmp_e', 'EX_3mb_e', 'EX_3ump_e', 'EX_acald_e', 'EX_adn_e', 'EX_ala__L_e', 'EX_arg__L_e', 'EX_asn__L_e', 'EX_asp__L_e', 'EX_btn_e', 'EX_cys__L_e', 'EX_fe3dcit_e', 'EX_glu__L_e', 'EX_his__L_e', 'EX_ile__L_e', 'EX_leu__L_e', 'EX_lys__L_e', 'EX_met__L_e', 'EX_mn2_e', 'EX_nac_e', 'EX_phe__L_e', 'EX_pro__L_e', 'EX_ribflv_e', 'EX_ser__L_e', 'EX_thm_e', 'EX_thr__L_e', 'EX_trp__L_e', 'EX_tyr__L_e', 'EX_ura_e', 'EX_val__L_e']
    
    # Try adding amino acids as a group
    with model:
        for rid in default_open:
            if rid in model.reactions: model.reactions.get_by_id(rid).lower_bound = -1000
        model.reactions.EX_glc__D_e.lower_bound = -10
        sol_full = model.optimize()
        print(f"Growth on Glucose + Complex supplements: {sol_full.objective_value}")

