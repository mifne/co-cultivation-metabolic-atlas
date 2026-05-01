import cobra
from cobra import Reaction, Metabolite
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def modify_ns21_model(input_path, output_path):
    logger.info(f"Loading draft model: {input_path}")
    model = cobra.io.read_sbml_model(input_path)
    
    # 1. Define ODTD (12-oxo-4,8-dimethyltrideca-4,8-diene-1-al, C15H24O2)
    # Extracellular and Intracellular
    odtd_e = Metabolite('odtd_e', formula='C15H24O2', name='ODTD', compartment='C_e')
    odtd_c = Metabolite('odtd_c', formula='C15H24O2', name='ODTD', compartment='C_c')
    
    # 2. Add Rubber Oxygenase (LatA1/A2 / RoxA/B)
    # 科学的シナジー: OR16が生成した rubber_fragment_e (C5H8O0.5) を NS21 が資化する
    # 3 Rubber fragments (C15H24O1.5) + 0.25 O2 -> 1 ODTD (C15H24O2)
    
    # 代謝物の取得ユーティリティ
    def get_met(met_id):
        for try_id in [met_id, f"M_{met_id}", f"M_{met_id}_c", f"M_{met_id}_e"]:
            if try_id in model.metabolites:
                return model.metabolites.get_by_id(try_id)
        raise KeyError(f"Metabolite {met_id} not found in model")

    if 'rubber_fragment_e' not in model.metabolites:
        rubber_fragment_e = Metabolite('M_rubber_fragment_e', formula='C5H8O0.5', name='Rubber fragment', compartment='C_e')
    else:
        rubber_fragment_e = model.metabolites.get_by_id('rubber_fragment_e')
        
    rxn_rox = Reaction('R_LATA1')
    rxn_rox.name = 'Rubber Oxygenase (LatA1/A2) - Fragment utilization'
    rxn_rox.add_metabolites({
        rubber_fragment_e: -3.0,
        get_met('o2_e'): -0.25,
        odtd_e: 1.0
    })
    rxn_rox.bounds = (0, 1000)
    
    # 3. ODTD Transport and Utilization
    # Transport
    rxn_odtdt = Reaction('ODTDt')
    rxn_odtdt.name = 'ODTD transport'
    rxn_odtdt.add_metabolites({odtd_e: -1.0, odtd_c: 1.0})
    rxn_odtdt.bounds = (0, 1000)
    
    # Utilization via Beta-oxidation (Simplified)
    # C15H24O2 -> 3 Acetyl-CoA + 3 Propionyl-CoA
    # We use metabolites from the model (standard IDs in CarveMe/BiGG)
    accoa = get_met('accoa_c')
    ppcoa = get_met('ppcoa_c')
    nad = get_met('nad_c')
    nadh = get_met('nadh_c')
    coa = get_met('coa_c')
    atp = get_met('atp_c')
    adp = get_met('adp_c')
    pi = get_met('pi_c')
    
    rxn_util = Reaction('ODTD_util')
    rxn_util.name = 'ODTD utilization to central metabolism'
    # Simplified stoichiometry: activation + cleavage
    rxn_util.add_metabolites({
        odtd_c: -1.0,
        atp: -1.0,
        coa: -6.0,
        nad: -4.0,
        accoa: 3.0,
        ppcoa: 3.0,
        nadh: 4.0,
        adp: 1.0,
        pi: 1.0
    })
    rxn_util.bounds = (0, 1000)
    
    # 4. PHA (PHB) Synthesis Pathway (if not already functional)
    # 2 Acetyl-CoA -> Acetoacetyl-CoA -> 3-hydroxybutyryl-CoA -> PHB
    # Check for existing reactions or add them
    if 'PHB_syn' not in [r.id for r in model.reactions]:
        # Simple aggregate reaction for simulation purposes
        # Acetyl-CoA + NADPH -> PHA + CoA + NADP
        # Add a placeholder PHA metabolite if missing
        if 'pha_c' not in model.metabolites:
            pha_c = Metabolite('pha_c', formula='C4H6O2', name='Polyhydroxyalkanoate', compartment='C_c')
        else:
            pha_c = model.metabolites.get_by_id('pha_c')
            
        rxn_pha = Reaction('PHB_syn')
        rxn_pha.name = 'PHB synthesis from Acetyl-CoA'
        rxn_pha.add_metabolites({
            accoa: -2.0,
            get_met('nadph_c'): -1.0,
            pha_c: 1.0, # Corrected: 1 unit of C4 (PHB unit) from 2 units of C2 (Acetyl-CoA)
            coa: 2.0,
            get_met('nadp_c'): 1.0
        })
        rxn_pha.bounds = (0, 1000)
        model.add_reactions([rxn_pha])
    
    # Add new components to model
    for r in [rxn_rox, rxn_odtdt, rxn_util]:
        if r.id in model.reactions:
            model.remove_reactions([model.reactions.get_by_id(r.id)])
        model.add_reactions([r])
    
    # 5. Fix Compartment IDs for COBRApy (same as OR16 fix)
    for reaction in model.reactions:
        if 'EX_' in reaction.id:
            for met in reaction.metabolites:
                met.compartment = 'C_e'
                
    # Create Exchange for Rubber fragment and ODTD
    if 'EX_rubber_fragment_e' not in [r.id for r in model.reactions]:
        rxn_ex_fragment = Reaction('EX_rubber_fragment_e')
        rxn_ex_fragment.add_metabolites({rubber_fragment_e: -1.0})
        rxn_ex_fragment.bounds = (0, 1000) # Controlled by simulator
        model.add_reactions([rxn_ex_fragment])
    
    if 'EX_odtd_e' not in [r.id for r in model.reactions]:
        rxn_ex_odtd = Reaction('EX_odtd_e')
        rxn_ex_odtd.add_metabolites({odtd_e: -1.0})
        rxn_ex_odtd.bounds = (0, 1000) # Secretion/Uptake
        model.add_reactions([rxn_ex_odtd])
    
    # 既存の EX_rubber_e は不要になるため閉じるか削除（任意だが、混乱を防ぐため閉じる）
    if 'EX_rubber_e' in model.reactions:
        model.reactions.EX_rubber_e.bounds = (0, 0)
    
    logger.info(f"Modified model: {len(model.reactions)} reactions, {len(model.metabolites)} metabolites")
    cobra.io.write_sbml_model(model, output_path)
    logger.info(f"Model saved to {output_path}")

if __name__ == "__main__":
    import sys
    # シナジー修復のため、既存の最終モデルをベースに修正を適用
    input_file = "models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml"
    output_file = "models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml"
    try:
        modify_ns21_model(input_file, output_file)
    except Exception as e:
        logger.error(f"Modification failed: {e}")
