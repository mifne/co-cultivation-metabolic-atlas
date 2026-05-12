import cobra
import pandas as pd
from pathlib import Path
from src.dfba_simulator import dFBASimulator

sbml_dir = Path("models/sbml/final_consortium")
models = {
    'OR16': cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml")),
    'NS21': cobra.io.read_sbml_model(str(sbml_dir / "Rhizobacter_gummiphilus_NS21.xml"))
}

# OR16 単体のテスト
sim_or16 = dFBASimulator(
    models={'OR16': models['OR16']},
    initial_biomass={'OR16': 0.5},
    initial_metabolites={'glc__D_e': 0.0, 'nh4_e': 10.0, 'pi_e': 10.0, 'yeast_extract_e': 1.0, 'C30_oligo_e': 0.1, 'h_e': 10**(-7.21) * 1000.0},
    initial_rubber=100.0,
    volume=1.0,
    dt=0.2
)

# NS21 単体のテスト
sim_ns21 = dFBASimulator(
    models={'NS21': models['NS21']},
    initial_biomass={'NS21': 0.5}, # OR16と同じ初期バイオマスで比較
    initial_metabolites={'glc__D_e': 0.0, 'nh4_e': 10.0, 'pi_e': 10.0, 'yeast_extract_e': 1.0, 'C30_oligo_e': 0.1, 'h_e': 10**(-7.21) * 1000.0},
    initial_rubber=100.0,
    volume=1.0,
    dt=0.2
)

def run_simulation(sim, name):
    records = []
    # 168時間 (840ステップ)
    for _ in range(840):
        state = sim.step({}, {'yeast_extract': 0.05, f'sn_{name.lower()}': 0.1}, dynamic_kla=100.0)
        records.append(state.rubber_concentration)
    return 100.0 - records[-1]

deg_or16 = run_simulation(sim_or16, 'OR16')
deg_ns21 = run_simulation(sim_ns21, 'NS21')

print(f"--- 168h Degradation (Single Species vs Co-culture) ---")
print(f"OR16 alone: {deg_or16:.2f} g")
print(f"NS21 alone: {deg_ns21:.2f} g")
