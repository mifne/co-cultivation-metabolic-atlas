import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")

models = {
    'OR16': cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml")),
    'NS21': cobra.io.read_sbml_model(str(sbml_dir / "Rhizobacter_gummiphilus_NS21.xml")),
    'LP': cobra.io.read_sbml_model(str(sbml_dir / "Lactobacillus_plantarum.xml"))
}

ye_components = set()

for name, model in models.items():
    current_set = [r.id for r in model.exchanges]
    essentials = []
    
    def get_mu(subset, m):
        with m:
            for r in m.exchanges: r.bounds = (0, 1000)
            for rid in subset: m.reactions.get_by_id(rid).bounds = (-1000, 1000)
            if 'EX_glc__D_e' in m.reactions: m.reactions.EX_glc__D_e.lower_bound = -10
            return m.optimize().objective_value

    for rid in current_set:
        test_set = [r for r in current_set if r != rid]
        if get_mu(test_set, model) <= 1e-6:
            essentials.append(rid)
    
    # Get met IDs
    for rid in essentials:
        r = model.reactions.get_by_id(rid)
        met = list(r.metabolites.keys())[0]
        mid = met.id
        while mid.startswith('M_'): mid = mid[2:]
        ye_components.add(mid)

# Remove standard ones
standard = ['glc__D_e', 'o2_e', 'h2o_e', 'h_e', 'nh4_e', 'pi_e']
final_ye = [c for c in ye_components if c not in standard]
print(f"FINAL_YE_LIST = {sorted(final_ye)}")
