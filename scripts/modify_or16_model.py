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

    # --- LCP遺伝子情報の準備 ---
    lcp_genes_info = {
        "ACTI_28730": "lcp1 (Latex clearing protein 1)",
        "ACTI_28740": "lcp2 (Latex clearing protein 2)",
        "ACTI_37800": "lcp3 (Latex clearing protein 3)"
    }

    # --- 代謝物の取得と定義 ---
    def get_met(met_id):
        # M_ プレフィックスの有無を両方試す
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

    rubber_e = get_or_create_metabolite('M_rubber_e', 'Natural rubber (polyisoprene)', 'C20H32', 'C_e')
    rubber_c = get_or_create_metabolite('M_rubber_c', 'Natural rubber (polyisoprene)', 'C20H32', 'C_c')
    aldehyde = get_or_create_metabolite('M_isoprenoid_aldehyde_c', 'Oligo-isoprenoid aldehyde', 'C20H32O', 'C_c')
    acid = get_or_create_metabolite('M_isoprenoid_acid_c', 'Oligo-isoprenoid acid', 'C20H32O2', 'C_c')
    rubber_fragment_e = get_or_create_metabolite('M_rubber_fragment_e', 'Rubber fragment', 'C5H8O', 'C_e')

    # --- 反応の取得または作成 ---
    
    # 1. Rubber transport
    if 'RUBBERt' in model.reactions:
        rxn_import = model.reactions.get_by_id('RUBBERt')
    else:
        rxn_import = Reaction('RUBBERt')
        rxn_import.name = 'Rubber transport'
        model.add_reactions([rxn_import])
    rxn_import.lower_bound = 0
    rxn_import.upper_bound = 1000
    rxn_import.add_metabolites({rubber_e: -1, rubber_c: 1}, combine=False)

    # 2. LCP Reaction
    if 'R_LCP' in model.reactions:
        rxn_lcp = model.reactions.get_by_id('R_LCP')
    else:
        rxn_lcp = Reaction('R_LCP')
        rxn_lcp.name = 'Latex Clearing Protein'
        model.add_reactions([rxn_lcp])
    
    o2_c = get_met('o2_c')
    rxn_lcp.add_metabolites({
        rubber_c: -1, 
        o2_c: -1, 
        aldehyde: 1, 
        rubber_fragment_e: 1
    }, combine=False)
    rxn_lcp.gene_reaction_rule = "ACTI_28730 or ACTI_28740 or ACTI_37800"
    rxn_lcp.lower_bound = 0
    rxn_lcp.upper_bound = 1000

    for gid, name in lcp_genes_info.items():
        if gid in model.genes:
            model.genes.get_by_id(gid).name = name

    # 3. Isoprenoid aldehyde dehydrogenase (OxiAB homolog)
    if 'ISOP_ALDH' in model.reactions:
        rxn_aldh = model.reactions.get_by_id('ISOP_ALDH')
    else:
        rxn_aldh = Reaction('ISOP_ALDH')
        rxn_aldh.name = 'Isoprenoid aldehyde dehydrogenase'
        model.add_reactions([rxn_aldh])
    
    # 科学的知見: OxiABはモリブデンヒドロキシラーゼであり、H2Oを酸素源とし、シトクロムcを電子受容体とする
    ficytc = get_met('ficytc_c') # Ferricytochrome c (oxidized)
    focytc = get_met('focytc_c') # Ferrocytochrome c (reduced)
    h2o = get_met('h2o_c')
    h = get_met('h_c')
    # 古い代謝物をクリア
    rxn_aldh.add_metabolites({m: -c for m, c in rxn_aldh.metabolites.items()})
    # 新しい量論をセット
    rxn_aldh.add_metabolites({
        aldehyde: -1, h2o: -1, ficytc: -2,
        acid: 1, focytc: 2, h: 2
    }, combine=False)
    rxn_aldh.lower_bound = 0
    rxn_aldh.upper_bound = 1000

    # 4. Isoprenoid acyl-CoA synthetase & Beta-oxidation (Lumped)
    if 'ISOP_ACS' in model.reactions:
        rxn_acs = model.reactions.get_by_id('ISOP_ACS')
    else:
        rxn_acs = Reaction('ISOP_ACS')
        rxn_acs.name = 'Isoprenoid acyl-CoA synthetase and beta-oxidation'
        model.add_reactions([rxn_acs])
    
    atp = get_met('atp_c')
    coa = get_met('coa_c')
    amp = get_met('amp_c')
    ppi = get_met('ppi_c')
    accoa = get_met('accoa_c')
    ppcoa = get_met('ppcoa_c')
    nad = get_met('nad_c')
    nadh = get_met('nadh_c')
    fad = get_met('fad_c')
    fadh2 = get_met('fadh2_c')
    
    # 古い代謝物をクリア
    rxn_acs.add_metabolites({m: -c for m, c in rxn_acs.metabolites.items()})
    
    # C20のベータ酸化の正確な質量・レドックスバランス:
    # 活性化で 1 ATP, 1 CoA -> 1 AMP, 1 PPi
    # 7回の切断サイクルで 7 CoA, 7 NAD+, 7 FAD -> 7 NADH, 7 FADH2, 7 H+
    # 生成物: 4 Acetyl-CoA + 4 Propionyl-CoA
    rxn_acs.add_metabolites({
        acid: -1, atp: -1, coa: -8, nad: -7, fad: -7,
        accoa: 4, ppcoa: 4, amp: 1, ppi: 1, nadh: 7, fadh2: 7, h: 7
    }, combine=False)
    rxn_acs.lower_bound = 0
    rxn_acs.upper_bound = 1000

    # 5. Rubber Exchange
    if 'EX_rubber_e' in model.reactions:
        rxn_ex = model.reactions.get_by_id('EX_rubber_e')
    else:
        rxn_ex = Reaction('EX_rubber_e')
        rxn_ex.name = 'Rubber exchange'
        model.add_reactions([rxn_ex])
    rxn_ex.add_metabolites({rubber_e: -1}, combine=False)
    rxn_ex.lower_bound = -1000
    rxn_ex.upper_bound = 1000

    # 保存
    cobra.io.write_sbml_model(model, output_path)
    logger.info(f"Modified model saved to {output_path}")

if __name__ == "__main__":
    target_file = "models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml"
    modify_or16_model(target_file, target_file)
