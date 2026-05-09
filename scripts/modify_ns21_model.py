import cobra
from cobra import Reaction, Metabolite
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def modify_ns21_model(input_path, output_path):
    logger.info(f"Loading NS21 model: {input_path}")
    model = cobra.io.read_sbml_model(input_path)
    
    def get_met(met_id):
        for try_id in [met_id, f"M_{met_id}"]:
            if try_id in model.metabolites:
                return model.metabolites.get_by_id(try_id)
        raise KeyError(f"Metabolite {met_id} not found in model")

    def get_or_create_metabolite(met_id, name, formula, compartment):
        try:
            return get_met(met_id)
        except KeyError:
            met = Metabolite(met_id, formula=formula, name=name, compartment=compartment)
            model.add_metabolites([met])
            return met

    # 1. 微生物学的定義に基づき代謝物を定義 (C30アーキテクチャ)
    rubber_bulk_e = get_or_create_metabolite('M_rubber_bulk_e', 'Rubber polymer unit (C5)', 'C5H8', 'C_e')
    c30_oligo_e = get_or_create_metabolite('M_C30_oligo_e', 'Oligo-isoprenoid (C30)', 'C30H48O2', 'C_e')
    c30_oligo_c = get_or_create_metabolite('M_C30_oligo_c', 'Oligo-isoprenoid (C30)', 'C30H48O2', 'C_c')
    odtd_e = get_or_create_metabolite('M_odtd_e', 'ODTD (C15)', 'C15H24O2', 'C_e')
    odtd_c = get_or_create_metabolite('M_odtd_c', 'ODTD (C15)', 'C15H24O2', 'C_c')
    o2_e = get_met('o2_e')

    # 2. R_ROXB (Endo-type): ポリマーを断片化してC30を作る
    if 'R_ROXB' in model.reactions:
        rxn_roxb = model.reactions.get_by_id('R_ROXB')
    else:
        rxn_roxb = Reaction('R_ROXB')
        model.add_reactions([rxn_roxb])
    
    rxn_roxb.name = 'Rubber Oxygenase RoxB (Endo-cleavage)'
    rxn_roxb.add_metabolites({m: -c for m, c in rxn_roxb.metabolites.items()})
    rxn_roxb.add_metabolites({rubber_bulk_e: -6.0, o2_e: -1.0, c30_oligo_e: 1.0}, combine=False)
    rxn_roxb.bounds = (0, 1000)

    # 3. R_ROXA (Exo-type): C30断片を基質として ODTD (C15) を切り出す
    # シナジーの核心: C30がないと動けない
    if 'R_ROXA' in model.reactions:
        rxn_roxa = model.reactions.get_by_id('R_ROXA')
    elif 'R_LATA1' in model.reactions: # 旧ID上書き
        rxn_roxa = model.reactions.get_by_id('R_LATA1')
        rxn_roxa.id = 'R_ROXA'
    else:
        rxn_roxa = Reaction('R_ROXA')
        model.add_reactions([rxn_roxa])
    
    rxn_roxa.name = 'Rubber Oxygenase RoxA (Exo-cleavage)'
    rxn_roxa.add_metabolites({m: -c for m, c in rxn_roxa.metabolites.items()})
    rxn_roxa.add_metabolites({c30_oligo_e: -1.0, o2_e: -1.0, odtd_e: 2.0}, combine=False)
    rxn_roxa.bounds = (0, 1000)

    # 4. Transport and Catabolism
    # ODTD Transport
    if 'R_ODTDt' in model.reactions:
        rxn_odtdt = model.reactions.get_by_id('R_ODTDt')
    elif 'ODTDt' in model.reactions:
        rxn_odtdt = model.reactions.get_by_id('ODTDt')
        rxn_odtdt.id = 'R_ODTDt'
    else:
        rxn_odtdt = Reaction('R_ODTDt')
        model.add_reactions([rxn_odtdt])
    rxn_odtdt.add_metabolites({m: -c for m, c in rxn_odtdt.metabolites.items()})
    rxn_odtdt.add_metabolites({odtd_e: -1.0, odtd_c: 1.0}, combine=False)
    rxn_odtdt.bounds = (0, 1000)

    # ODTD Catabolism
    if 'R_ODTD_cat' in model.reactions:
        rxn_odtd_cat = model.reactions.get_by_id('R_ODTD_cat')
    elif 'ODTD_util' in model.reactions:
        rxn_odtd_cat = model.reactions.get_by_id('ODTD_util')
        rxn_odtd_cat.id = 'R_ODTD_cat'
    else:
        rxn_odtd_cat = Reaction('R_ODTD_cat')
        model.add_reactions([rxn_odtd_cat])
    
    coa = get_met('coa_c')
    h2o = get_met('h2o_c')
    nad = get_met('nad_c')
    fad = get_met('fad_c')
    atp = get_met('atp_c')
    accoa = get_met('accoa_c')
    ppcoa = get_met('ppcoa_c')
    nadh = get_met('nadh_c')
    fadh2 = get_met('fadh2_c')
    h = get_met('h_c')
    amp = get_met('amp_c')
    ppi = get_met('ppi_c')

    rxn_odtd_cat.add_metabolites({m: -c for m, c in rxn_odtd_cat.metabolites.items()})
    # C15H24O2 + 6 CoA + 4 H2O + 2 NAD + 2 FAD + 1 ATP -> 3 ACCOA + 3 PPCOA + 2 NADH + 2 FADH2 + 2 H + 1 AMP + 1 PPi
    rxn_odtd_cat.add_metabolites({
        odtd_c: -1.0,
        coa: -6.0,
        h2o: -4.0,
        nad: -2.0,
        fad: -2.0,
        atp: -1.0,
        accoa: 3.0,
        ppcoa: 3.0,
        nadh: 2.0,
        fadh2: 2.0,
        h: 2.0,
        amp: 1.0,
        ppi: 1.0
    }, combine=False)
    rxn_odtd_cat.bounds = (0, 1000)

    # (Option) NS21 C30 utilization
    if 'R_C30t' not in model.reactions:
        rxn_c30t = Reaction('R_C30t')
        rxn_c30t.add_metabolites({c30_oligo_e: -1.0, c30_oligo_c: 1.0})
        # Block direct C30 uptake to force Exo-cleavage (RoxA)
        rxn_c30t.bounds = (0, 0.0)
        model.add_reactions([rxn_c30t])
    else:
        model.reactions.get_by_id('R_C30t').bounds = (0, 0.0)
    
    if 'R_C30_cat' not in model.reactions:
        rxn_c30_cat = Reaction('R_C30_cat')
        rxn_c30_cat.add_metabolites({
            c30_oligo_c: -1.0, coa: -12.0, h2o: -10.0, nad: -5.0, fad: -5.0, atp: -1.0,
            accoa: 6.0, ppcoa: 6.0, nadh: 5.0, fadh2: 5.0, h: 5.0, amp: 1.0, ppi: 1.0
        })
        rxn_c30_cat.bounds = (0, 1000)
        model.add_reactions([rxn_c30_cat])

    # 5. Exchange reactions
    for met, rid in [(rubber_bulk_e, 'EX_rubber_bulk_e'), (c30_oligo_e, 'EX_C30_oligo_e'), (odtd_e, 'EX_odtd_e')]:
        if rid in model.reactions:
            rxn = model.reactions.get_by_id(rid)
        else:
            rxn = Reaction(rid)
            model.add_reactions([rxn])
        rxn.add_metabolites({m: -c for m, c in rxn.metabolites.items()})
        rxn.add_metabolites({met: -1.0}, combine=False)
        rxn.bounds = (-1000, 1000)

    # 既存の EX_rubber_e, EX_rubber_fragment_e はクリーンアップ
    for old_rxn in ['EX_rubber_e', 'EX_rubber_fragment_e']:
        if old_rxn in model.reactions:
            model.reactions.get_by_id(old_rxn).bounds = (0, 0)

    logger.info(f"Modified NS21 model (v3): {len(model.reactions)} reactions")
    cobra.io.write_sbml_model(model, output_path)
    logger.info(f"Model saved to {output_path}")

if __name__ == "__main__":
    input_file = "models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml"
    modify_ns21_model(input_file, input_file)
