import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import cobra
import os

def prepare_specific_nutrients():
    sbml_dir = 'models/sbml/final_consortium'
    
    # 1. OR16: Maltotriose (mlttr_e)
    or16 = cobra.io.read_sbml_model(os.path.join(sbml_dir, 'Actinoplanes_sp_OR16_lcp.xml'))
    if 'mlttr_e' not in or16.metabolites:
        met = cobra.Metabolite('M_mlttr_e', formula='C18H32O16', name='Maltotriose', compartment='C_e')
        rxn = cobra.Reaction('EX_mlttr_e')
        rxn.add_metabolites({met: -1})
        rxn.bounds = (-1000, 1000)
        or16.add_reactions([rxn])
        # 分解経路（マルトトリオース -> 3 グルコース）を簡易的に追加
        glc = or16.metabolites.get_by_id('M_glc__D_e') if 'M_glc__D_e' in or16.metabolites else or16.metabolites.glc__D_e
        hydrolysis = cobra.Reaction('MLTTR_hydrolysis')
        hydrolysis.add_metabolites({met: -1, or16.metabolites.h2o_c: -2, glc: 3})
        hydrolysis.bounds = (0, 1000)
        or16.add_reactions([hydrolysis])
    cobra.io.write_sbml_model(or16, os.path.join(sbml_dir, 'Actinoplanes_sp_OR16_lcp.xml'))

    # 2. NS21: Putrescine (ptrc_e)
    ns21 = cobra.io.read_sbml_model(os.path.join(sbml_dir, 'Rhizobacter_gummiphilus_NS21.xml'))
    if 'ptrc_e' not in ns21.metabolites:
        met = cobra.Metabolite('M_ptrc_e', formula='C4H12N2', name='Putrescine', compartment='C_e')
        rxn = cobra.Reaction('EX_ptrc_e')
        rxn.add_metabolites({met: -1})
        rxn.bounds = (-1000, 1000)
        ns21.add_reactions([rxn])
        # NS21には既に代謝経路がある可能性が高いが、念のため取り込みを保証
    cobra.io.write_sbml_model(ns21, os.path.join(sbml_dir, 'Rhizobacter_gummiphilus_NS21.xml'))

    # 3. LP: Mannitol (mnl_e)
    lp = cobra.io.read_sbml_model(os.path.join(sbml_dir, 'Lactobacillus_plantarum.xml'))
    # LPは通常マンニトールを資化できる。
    if 'EX_mnl_e' not in [r.id for r in lp.reactions]:
        met = cobra.Metabolite('M_mnl_e', formula='C6H14O6', name='D-Mannitol', compartment='e')
        rxn = cobra.Reaction('EX_mnl_e')
        rxn.add_metabolites({met: -1})
        rxn.bounds = (-1000, 1000)
        lp.add_reactions([rxn])
    cobra.io.write_sbml_model(lp, os.path.join(sbml_dir, 'Lactobacillus_plantarum.xml'))

    print("✅ Models updated with species-specific nutrients (Maltotriose, Putrescine, Mannitol).")

if __name__ == "__main__":
    prepare_specific_nutrients()
