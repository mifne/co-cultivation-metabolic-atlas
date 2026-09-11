import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import cobra
from pathlib import Path
sbml_dir = Path("models/sbml/final_consortium")
model = cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))

essentials = ['EX_ca2_e', 'EX_cl_e', 'EX_cobalt2_e', 'EX_cu_e', 'EX_k_e', 'EX_mg2_e', 'EX_mn2_e', 'EX_o2_e', 'EX_salchs4fe_e', 'EX_tsul_e', 'EX_ump_e', 'EX_xtsn_e', 'EX_zn2_e']

with model:
    for r in model.exchanges: r.lower_bound = 0
    for rid in essentials: model.reactions.get_by_id(rid).lower_bound = -1000
    
    # Try Glucose
    model.reactions.EX_glc__D_e.lower_bound = -10
    print(f"Growth with Glucose: {model.optimize().objective_value}")

