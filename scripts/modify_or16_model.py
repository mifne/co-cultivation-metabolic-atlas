import cobra
from cobra import Model, Reaction, Metabolite, Gene
import logging
import os

def modify_or16_model(input_path, output_path):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist.")
        return

    try:
        model = cobra.io.read_sbml_model(input_path)
        logger.info(f"Loaded model: {model.id}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # --- 代謝物の取得と定義 ---
    def get_met(met_id):
        for attempt in [met_id, f"M_{met_id}"]:
            try:
                return model.metabolites.get_by_id(attempt)
            except KeyError:
                continue
        raise KeyError(f"Metabolite {met_id} not found in model")

    def get_or_create_metabolite(met_id, name, formula, compartment):
        try:
            return get_met(met_id)
        except KeyError:
            met = Metabolite(met_id, formula=formula, name=name, compartment=compartment)
            model.add_metabolites([met])
            return met

    # 1. 微生物学的定義に基づき代謝物を定義 (C30アーキテクチャ)
    # C5ユニットとしてのバルクゴム
    rubber_bulk_e = get_or_create_metabolite('M_rubber_bulk_e', 'Rubber polymer unit (C5)', 'C5H8', 'C_e')
    # 中鎖オリゴマー (C30)
    c30_oligo_e = get_or_create_metabolite('M_C30_oligo_e', 'Oligo-isoprenoid (C30)', 'C30H48O2', 'C_e')
    c30_oligo_c = get_or_create_metabolite('M_C30_oligo_c', 'Oligo-isoprenoid (C30)', 'C30H48O2', 'C_c')

    # 不要な古い代謝物のクリーンアップ (もしあれば)
    for old_id in ['M_rubber_e', 'M_rubber_c', 'M_isoprenoid_aldehyde_c', 'M_isoprenoid_acid_c', 'M_rubber_fragment_e']:
        if old_id in model.metabolites:
            # 関連する反応も消す必要があるが、下で Reaction ID を再利用・上書きする
            pass

    # 2. R_LCP (Endo-type): 巨大ポリマーを断片化してC30を作る
    if 'R_LCP' in model.reactions:
        rxn_lcp = model.reactions.get_by_id('R_LCP')
    else:
        rxn_lcp = Reaction('R_LCP')
        model.add_reactions([rxn_lcp])
    
    rxn_lcp.name = 'Latex Clearing Protein (Endo-cleavage)'
    o2_e = get_met('o2_e')
    # 完全に新しい量論をセット
    rxn_lcp.add_metabolites({m: -c for m, c in rxn_lcp.metabolites.items()})
    rxn_lcp.add_metabolites({
        rubber_bulk_e: -6.0, 
        o2_e: -1.0, 
        c30_oligo_e: 1.0
    }, combine=False)
    rxn_lcp.gene_reaction_rule = "ACTI_28730 or ACTI_28740 or ACTI_37800"
    rxn_lcp.lower_bound = 0
    rxn_lcp.upper_bound = 1000

    # 3. C30 Transport
    if 'R_C30t' in model.reactions:
        rxn_t = model.reactions.get_by_id('R_C30t')
    else:
        rxn_t = Reaction('R_C30t')
        model.add_reactions([rxn_t])
    rxn_t.name = 'C30 oligomer transport'

    atp = get_met('atp_c')
    adp = get_met('adp_c')
    pi = get_met('pi_c')
    h2o = get_met('h2o_c')
    h = get_met('h_c')

    rxn_t.add_metabolites({m: -c for m, c in rxn_t.metabolites.items()})
    # ABC transporter mechanism: 1 ATP per molecule transported
    rxn_t.add_metabolites({
        c30_oligo_e: -1.0,
        atp: -1.0,
        h2o: -1.0,
        c30_oligo_c: 1.0,
        adp: 1.0,
        pi: 1.0,
        h: 1.0
    }, combine=False)
    rxn_t.lower_bound = 0
    # Membrane permeability limit for large hydrophobic oligomer (C30)
    # Typical lipid/fatty acid uptake Vmax in bacteria is ~0.5 - 2.0 mmol/gDW/h
    rxn_t.upper_bound = 0.5

    # 4. C30 Catabolism (Beta-oxidation connection)
    # 指示に基づき 6 Acetyl-CoA + 6 Propionyl-CoA を生成
    if 'R_C30_cat' in model.reactions:
        rxn_cat = model.reactions.get_by_id('R_C30_cat')
    elif 'ISOP_ACS' in model.reactions: # 旧IDを上書き
        rxn_cat = model.reactions.get_by_id('ISOP_ACS')
        rxn_cat.id = 'R_C30_cat'
    else:
        rxn_cat = Reaction('R_C30_cat')
        model.add_reactions([rxn_cat])
    
    rxn_cat.name = 'C30 catabolism via beta-oxidation'
    
    # 必要な補助代謝物
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

    rxn_cat.add_metabolites({m: -c for m, c in rxn_cat.metabolites.items()})
    # 精緻な量論: 
    # C30H48O2 + 12 CoA + 10 H2O + 5 NAD + 5 FAD + 1 ATP -> 6 ACCOA + 6 PPCOA + 5 NADH + 5 FADH2 + 5 H + 1 AMP + 1 PPi
    rxn_cat.add_metabolites({
        c30_oligo_c: -1.0,
        coa: -12.0,
        h2o: -10.0,
        nad: -5.0,
        fad: -5.0,
        atp: -1.0,
        accoa: 6.0,
        ppcoa: 6.0,
        nadh: 5.0,
        fadh2: 5.0,
        h: 5.0,
        amp: 1.0,
        ppi: 1.0
    }, combine=False)
    rxn_cat.lower_bound = 0
    rxn_cat.upper_bound = 1000

    # 5. Exchange reactions
    # rubber_bulk_e の Exchange (Simulatorで減少させる対象)
    if 'EX_rubber_bulk_e' in model.reactions:
        rxn_ex = model.reactions.get_by_id('EX_rubber_bulk_e')
    else:
        rxn_ex = Reaction('EX_rubber_bulk_e')
        model.add_reactions([rxn_ex])
    rxn_ex.name = 'Rubber bulk exchange (C5)'
    rxn_ex.add_metabolites({m: -c for m, c in rxn_ex.metabolites.items()})
    rxn_ex.add_metabolites({rubber_bulk_e: -1.0}, combine=False)
    rxn_ex.lower_bound = -1000 # 基質として取り込み可能
    rxn_ex.upper_bound = 1000

    # C30 exchange
    if 'EX_C30_oligo_e' in model.reactions:
        rxn_ex_c30 = model.reactions.get_by_id('EX_C30_oligo_e')
    else:
        rxn_ex_c30 = Reaction('EX_C30_oligo_e')
        model.add_reactions([rxn_ex_c30])
    rxn_ex_c30.add_metabolites({m: -c for m, c in rxn_ex_c30.metabolites.items()})
    rxn_ex_c30.add_metabolites({c30_oligo_e: -1.0}, combine=False)
    rxn_ex_c30.bounds = (-1000, 1000)

    # 不要な古い反応のクリーンアップ
    for old_rxn in ['RUBBERt', 'ISOP_ALDH', 'EX_rubber_e', 'EX_rubber_fragment_e']:
        if old_rxn in model.reactions:
            model.remove_reactions([model.reactions.get_by_id(old_rxn)])

    # 保存
    cobra.io.write_sbml_model(model, output_path)
    logger.info(f"Modified OR16 model (v3) saved to {output_path}")

if __name__ == "__main__":
    target_file = "models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml"
    modify_or16_model(target_file, target_file)
