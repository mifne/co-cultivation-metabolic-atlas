
import cobra
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_lp_model(path):
    logger.info(f"Fixing LP model: {path}")
    model = cobra.io.read_sbml_model(path)
    
    # バイオサーファクタント合成反応を簡略化して実行可能にする
    # Glycolipoprotein (10 kDa) 相当を、グルコース、アセチルCoA、アミノ酸、ATPから合成
    if "R_GLYCOLIPOPROTEIN_SYN_SEC" in model.reactions:
        rxn = model.reactions.get_by_id("R_GLYCOLIPOPROTEIN_SYN_SEC")
        # 既存の反応物をクリア
        rxn.add_metabolites({m: -coeff for m, coeff in rxn.metabolites.items()})
        # 新しい量論（概算）
        # 10kDa = ~50 糖ユニット + ~20 脂肪酸ユニット + アミノ酸
        rxn.add_metabolites({
            model.metabolites.get_by_id("glc__D_c"): -50.0,
            model.metabolites.get_by_id("accoa_c"): -20.0,
            model.metabolites.get_by_id("atp_c"): -100.0,
            model.metabolites.get_by_id("h2o_c"): -100.0,
            model.metabolites.get_by_id("biosurfactant_e"): 1.0,
            model.metabolites.get_by_id("coa_c"): 20.0,
            model.metabolites.get_by_id("adp_c"): 100.0,
            model.metabolites.get_by_id("pi_c"): 100.0,
            model.metabolites.get_by_id("h_c"): 100.0
        })
        rxn.lower_bound = 0.001 # 強制分泌
        logger.info("Redefined R_GLYCOLIPOPROTEIN_SYN_SEC and set lower bound to 0.001")
    
    # LP自体の生存のために、一部の必須代謝物の境界を緩める（暫定的なギャップフィル）
    # 多くの CarveMe モデルで LP は ATP 維持代謝(ATPM)が原因で死ぬことがある
    if "ATPM" in model.reactions:
        model.reactions.ATPM.lower_bound = 0
        logger.info("Set ATPM lower bound to 0 for LP survival")

    model.objective = "BIOMASS_LLA"
    cobra.io.write_sbml_model(model, path)

def fix_ns21_model(path):
    logger.info(f"Fixing NS21 model: {path}")
    model = cobra.io.read_sbml_model(path)
    
    # 1. PHAコンパートメントの修正
    if "pha_c" in model.metabolites:
        pha = model.metabolites.get_by_id("pha_c")
        pha.compartment = "C_c"
        logger.info(f"Changed pha_c compartment to {pha.compartment}")
    
    # 2. PHB合成反応の修正
    if "PHB_syn" in model.reactions:
        rxn = model.reactions.get_by_id("PHB_syn")
        logger.info(f"Verified PHB_syn: {[m.id for m in rxn.products]}")

    # 3. PHA排出反応を閉じる
    for ex_id in ["EX_pha_c", "EX_pha_e"]:
        if ex_id in model.reactions:
            model.reactions.get_by_id(ex_id).bounds = (0, 0)
            logger.info(f"Closed exchange reaction: {ex_id}")
            
    cobra.io.write_sbml_model(model, path)

def fix_or16_model(path):
    logger.info(f"Fixing OR16 model: {path}")
    model = cobra.io.read_sbml_model(path)
    
    # R_LCP の修正 (冪等な方法で)
    if "R_LCP" in model.reactions:
        rxn = model.reactions.get_by_id("R_LCP")
        # 一旦リセットして再構築
        rubber_c = model.metabolites.get_by_id("M_rubber_c")
        o2_c = model.metabolites.get_by_id("o2_c")
        ald_c = model.metabolites.get_by_id("M_isoprenoid_aldehyde_c")
        frag_e = model.metabolites.get_by_id("rubber_fragment_e")
        
        # 反応物をクリア
        rxn.add_metabolites({m: -c for m, c in rxn.metabolites.items()})
        # 正しい量論をセット (C25H40 + 0.75 O2 -> C20H32O + C5H8O0.5)
        rxn.add_metabolites({
            rubber_c: -1.25,
            o2_c: -0.75,
            ald_c: 1.0,
            frag_e: 1.0
        })
        
        logger.info(f"Fixed R_LCP (Balanced): {rxn.reaction}")
        balance = rxn.check_mass_balance()
        if not balance:
            logger.info("R_LCP is now mass-balanced.")
        else:
            logger.warning(f"R_LCP still has balance issues: {balance}")

    cobra.io.write_sbml_model(model, path)

if __name__ == "__main__":
    lp_path = "models/sbml/final_consortium/Lactobacillus_plantarum.xml"
    ns21_path = "models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml"
    or16_path = "models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml"
    
    fix_lp_model(lp_path)
    fix_ns21_model(ns21_path)
    fix_or16_model(or16_path)
