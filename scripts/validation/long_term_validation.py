import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import cobra
import numpy as np
import pandas as pd
from pathlib import Path
from src.dfba_simulator import dFBASimulator

sbml_dir = Path("models/sbml/final_consortium")
models = {
    'OR16': cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml")),
    'NS21': cobra.io.read_sbml_model(str(sbml_dir / "Rhizobacter_gummiphilus_NS21.xml")),
    'LP': cobra.io.read_sbml_model(str(sbml_dir / "Lactobacillus_plantarum.xml"))
}

# Initial metabolites with 1.0 YE
initial_metabolites = {
    'glc__D_e': 0.0, 'nh4_e': 10.0, 'pi_e': 10.0, 'yeast_extract_e': 1.0,
    'o2_e': 0.25, 'h2o_e': 55000.0, 'h_e': 1e-4
}

sim = dFBASimulator(
    models=models,
    initial_biomass={'OR16': 0.5, 'NS21': 0.1, 'LP': 0.1},
    initial_metabolites=initial_metabolites,
    dt=0.2
)

history = []
# Simulate 168 hours (840 steps at dt=0.2)
for i in range(840):
    # Static "good" actions: small YE and some kla
    state = sim.step({}, {'yeast_extract': 0.1}, dynamic_kla=50.0)
    h_e = state.metabolites.get('h_e', 1e-4)
    ph = -np.log10(h_e/1000.0)
    
    history.append({
        'time': state.time,
        'rubber': state.rubber_concentration,
        'ph': ph,
        'biomass_or16': state.species['OR16'].biomass,
        'biomass_ns21': state.species['NS21'].biomass,
        'c30': state.metabolites.get('C30_oligo_e', 0.0),
        'odtd': state.metabolites.get('odtd_e', 0.0)
    })

df = pd.DataFrame(history)
print(f"--- 168h Simulation Results ---")
print(f"Final Rubber: {df['rubber'].iloc[-1]:.4f} (Decrease: {100.0 - df['rubber'].iloc[-1]:.4f}%)")
print(f"Max Biomass OR16: {df['biomass_or16'].max():.4f}")
print(f"Final pH: {df['ph'].iloc[-1]:.4f}")
print(f"Final C30: {df['c30'].iloc[-1]:.4e}")
print(f"Final ODTD: {df['odtd'].iloc[-1]:.4e}")

# Check if rubber ever decreased significantly
if df['rubber'].min() < 100.0:
    print("SUCCESS: Rubber degradation is physically occurring in the simulator.")
else:
    print("FAILURE: Rubber is NOT degrading even in 168h.")

