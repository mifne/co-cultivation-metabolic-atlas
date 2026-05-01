import cobra
from cobra import Reaction, Metabolite

def add_scavenging_to_lp():
    model_path = 'models/sbml/final_consortium/Lactobacillus_plantarum.xml'
    model = cobra.io.read_sbml_model(model_path)
    
    # 1. 外部フラグメントの定義
    if 'rubber_fragment_e' not in model.metabolites:
        met = Metabolite('rubber_fragment_e', name='Rubber Fragment', compartment='C_e')
        model.add_metabolites([met])
    
    # 2. 交換反応の追加
    if 'EX_rubber_fragment_e' not in model.reactions:
        rxn = Reaction('EX_rubber_fragment_e')
        rxn.name = 'Rubber fragment exchange'
        rxn.add_metabolites({model.metabolites.rubber_fragment_e: -1})
        rxn.lower_bound = -10.0 # 摂取可能
        rxn.upper_bound = 1000.0
        model.add_reactions([rxn])
        
    # 3. 代謝バイパス (Scavenging: Fragment -> Glucose-6-phosphate)
    # C5 (Fragment) -> 5/6 C6 (G6P)
    if 'RUBBER_SCAVENGING' not in model.reactions:
        rxn = Reaction('RUBBER_SCAVENGING')
        rxn.name = 'Rubber fragment scavenging bypass'
        # LPの解糖系中間体を探す (g6p_c 等)
        g6p = None
        for m in model.metabolites:
            if 'g6p_c' in m.id:
                g6p = m
                break
        
        if g6p:
            rxn.add_metabolites({
                model.metabolites.rubber_fragment_e: -1.0,
                g6p: 0.833
            })
            rxn.lower_bound = 0
            rxn.upper_bound = 1000.0
            model.add_reactions([rxn])
            print(f"✅ Added scavenging: Rubber Fragment -> {g6p.id}")
        else:
            print("❌ Could not find G6P in LP model.")

    cobra.io.write_sbml_model(model, model_path)

if __name__ == "__main__":
    add_scavenging_to_lp()
