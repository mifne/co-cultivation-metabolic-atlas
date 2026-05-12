import cobra
import pandas as pd
from pathlib import Path
from src.dfba_simulator import dFBASimulator

sbml_dir = Path("models/sbml/final_consortium")
models = {
    'OR16': cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml"))
}

# OR16 単体のテスト (pHを中性に固定する簡易設定)
sim_or16 = dFBASimulator(
    models={'OR16': models['OR16']},
    initial_biomass={'OR16': 0.5},
    initial_metabolites={'glc__D_e': 0.0, 'nh4_e': 10.0, 'pi_e': 10.0, 'yeast_extract_e': 1.0, 'C30_oligo_e': 0.1, 'h_e': 10**(-7.0) * 1000.0},
    initial_rubber=100.0,
    volume=1.0,
    dt=0.2
)

# pHが極端に変動して死なないように、強制的にpH7付近を維持する
for _ in range(840):
    sim_or16.state.metabolites['h_e'] = 10**(-7.0) * 1000.0
    state = sim_or16.step({}, {'yeast_extract': 0.05, 'sn_or16': 0.1}, dynamic_kla=100.0)

print(f"OR16 alone (pH clamped to 7.0): {100.0 - state.rubber_concentration:.2f} g")
print(f"Final OR16 Biomass: {state.species['OR16'].biomass:.2f}")

