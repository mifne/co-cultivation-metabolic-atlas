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

print("--- Initial State ---")
print(f"Rubber: {env.simulator.state.rubber_concentration}")

# Simulate 5 steps with random action
for i in range(5):
    # Action [OR16, NS21, LP, YE, KLa]
    # Let's try action where YE is 0
    action = np.array([0.1, 0.1, 0.1, 0.0, 0.5], dtype=np.float32)
    obs, reward, term, trunc, info = env.step(action)
    print(f"Step {i+1}: Rubber={info['rubber_remaining']:.6f}, pH={info['ph']:.2f}, OR16={info['biomass_or16']:.4f}")

