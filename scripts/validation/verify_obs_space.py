import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import cobra
import numpy as np
from pathlib import Path
from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv

sbml_dir = Path("models/sbml/final_consortium")
models = {
    'OR16': cobra.io.read_sbml_model(str(sbml_dir / "Actinoplanes_sp_OR16_lcp.xml")),
    'NS21': cobra.io.read_sbml_model(str(sbml_dir / "Rhizobacter_gummiphilus_NS21.xml")),
    'LP': cobra.io.read_sbml_model(str(sbml_dir / "Lactobacillus_plantarum.xml"))
}

sim = dFBASimulator(
    models=models,
    initial_biomass={'OR16': 0.5, 'NS21': 0.1, 'LP': 0.1},
    initial_metabolites={'glc__D_e': 0.0, 'nh4_e': 10.0, 'pi_e': 10.0, 'yeast_extract_e': 1.0},
    dt=0.2
)

env = ConsortiumEnv(simulator=sim)
obs, _ = env.reset()

print(f"Observation shape: {obs.shape}")
# Should be (16,) = 3 (biomass) + 13 (others)
if obs.shape[0] == 16:
    print("SUCCESS: Observation space expanded correctly.")
else:
    print(f"FAILURE: Expected 16, got {obs.shape[0]}")

# Print obs to check values
print(f"Observation values: {obs}")

# Check if C30 is in state after reset
c30 = env.simulator.state.metabolites.get('C30_oligo_e', 0.0)
print(f"C30_oligo_e after reset: {c30}")
if c30 > 0:
    print("SUCCESS: C30 inducer seed found.")
else:
    print("FAILURE: C30 inducer seed missing.")

