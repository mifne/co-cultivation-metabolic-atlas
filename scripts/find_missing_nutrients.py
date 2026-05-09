import cobra
from pathlib import Path

def find_missing_nutrients():
    sbml_dir = Path('models/sbml/final_consortium')
    for species in ['Actinoplanes_sp_OR16_lcp.xml', 'Rhizobacter_gummiphilus_NS21.xml']:
        model_path = sbml_dir / species
        model = cobra.io.read_sbml_model(str(model_path))
        print(f"\n--- Model: {species} ---")
        
        # 開放された交換反応のリストを取得
        exchanges = [rxn.id for rxn in model.exchanges]
        
        # 1. 緩い制約で増殖可能か？
        with model:
            for rxn in model.exchanges:
                rxn.lower_bound = -1000.0
            sol = model.optimize()
            print(f"  Unconstrained Growth: {sol.objective_value:.6f}")
            
            if sol.objective_value < 1e-6:
                print("  ERROR: Even with all exchanges open, Biomass is 0!")
                continue
            
            # 2. 必須交換反応の特定 (Minimal Medium)
            from cobra.medium import minimal_medium
            min_med = minimal_medium(model, 0.1)
            if min_med is not None:
                print(f"  Minimal Medium required for growth (at mu=0.1):")
                for rxn_id, flux in min_med.items():
                    met_names = [m.name for m in model.reactions.get_by_id(rxn_id).metabolites]
                    print(f"    {rxn_id}: {met_names}")
            else:
                print("  Failed to find minimal medium.")

if __name__ == "__main__":
    find_missing_nutrients()
