import cobra
from main import load_sbml_models, select_consortium_models, get_initial_params
from pathlib import Path

sbml_dir = Path('models/sbml/final_consortium')
all_models = load_sbml_models(sbml_dir)
models = select_consortium_models(all_models)
ns21 = models['Rhizobacter_gummiphilus_NS21']

# Let's simulate what dfba_simulator does for NS21 at t=0
# initial_metabolites = get_initial_params(models)[1]
initial_metabolites = {
    'glc__D_e': 0.5, 'nh4_e': 50.0, 'pi_e': 50.0, 'o2_e': 0.25, 
    'so4_e': 2.0, 'mg2_e': 2.0, 'ca2_e': 0.1, 'k_e': 10.0, 'cl_e': 10.0,
    'fe3_e': 0.1, 'fe2_e': 0.1, 'h_e': 0.0001, 'h2o_e': 55000.0, 'co2_e': 1.0,
    'zn2_e': 0.01, 'mn2_e': 0.1, 'cu2_e': 0.01, 'cobalt2_e': 0.01, 
    'ni2_e': 0.01, 'mobd_e': 0.01,
    'nac_e': 0.1, 'ribflv_e': 0.1, 'pnto__R_e': 0.1, 'thm_e': 0.1, 
    'btn_e': 0.1, '4abz_e': 0.1, 'fol_e': 0.1, 'nicnt_e': 0.1,
    'ade_e': 0.1, 'gua_e': 0.1, 'ura_e': 0.1, 'xan_e': 0.1, 'orot_e': 0.1,
    'ins_e': 0.1, 'thymd_e': 0.1,
    'ala__L_e': 1.0, 'arg__L_e': 1.0, 'asn__L_e': 1.0, 'asp__L_e': 1.0, 
    'cys__L_e': 1.0, 'gln__L_e': 1.0, 'glu__L_e': 1.0, 'gly_e': 1.0, 
    'his__L_e': 1.0, 'ile__L_e': 1.0, 'leu__L_e': 1.0, 'lys__L_e': 1.0, 
    'met__L_e': 1.0, 'phe__L_e': 1.0, 'pro__L_e': 1.0, 'ser__L_e': 1.0, 
    'thr__L_e': 1.0, 'trp__L_e': 1.0, 'tyr__L_e': 1.0, 'val__L_e': 1.0,
    'ptrc_e': 5.0, # sn_ns21 action adds this
    'rubber_fragment_e': 0.1
}

for rxn in ns21.exchanges:
    rxn.lower_bound = 0

for met_id, conc in initial_metabolites.items():
    if conc > 1e-6:
        # find exchange rxn
        rid = None
        for rxn in ns21.exchanges:
            for met in rxn.metabolites:
                clean_id = met.id[2:] if met.id.startswith('M_') else met.id
                if clean_id == met_id:
                    rid = rxn.id
                    break
        if rid:
            # v_max logic
            rxn = ns21.reactions.get_by_id(rid)
            Km = 0.01 if met_id in ['glc__D_e', 'o2_e', 'pi_e', 'nh4_e'] else 0.1
            v_max = 10.0
            rxn.lower_bound = -v_max * (conc / (Km + conc))

# rubber
ns21.reactions.get_by_id('EX_rubber_bulk_e').lower_bound = -1000

# h_e bound in dfba_simulator:
# if h_id in self.exchange_reactions[species_name]:
#     model.reactions.get_by_id(rxn_id).lower_bound = -1000.0
# Let's set it to -1000
h_rxn = [r.id for r in ns21.exchanges if "h_e" in r.id][0]
ns21.reactions.get_by_id(h_rxn).lower_bound = -1000

sol = ns21.optimize()
print("Growth:", sol.objective_value)
for r in ns21.reactions:
    if "EX_" in r.id and abs(sol.fluxes.get(r.id, 0)) > 1:
        print(f"{r.id}: {sol.fluxes.get(r.id, 0):.2f}")
