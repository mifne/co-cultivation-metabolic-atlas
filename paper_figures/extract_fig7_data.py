import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import load_sbml_models, select_consortium_models, get_initial_params
from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv

def make_env_custom(sbml_dir, env_params, dt=0.2):
    all_models = load_sbml_models(Path(sbml_dir))
    models = select_consortium_models(all_models)
    initial_biomass, initial_metabolites = get_initial_params(models)
    sim = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=100.0,
        volume=1.0,
        dt=dt
    )
    return ConsortiumEnv(simulator=sim, **env_params)

class PIController:
    def __init__(self, target_ph=7.0, kp=1.0, ki=0.1):
        self.target_ph = target_ph
        self.kp = kp
        self.ki = ki
        self.integral_error = 0.0
        
    def get_action(self, current_ph):
        error = current_ph - self.target_ph
        self.integral_error += error
        self.integral_error = max(-10.0, min(10.0, self.integral_error))
        u = self.kp * error + self.ki * self.integral_error
        sn_lp = max(0.0, min(1.0, u))
        return np.array([0.05, 0.05, sn_lp, 0.05, 0.5])

def run_shock_simulation(controller_type, model_path, vec_path, sbml_dir):
    dt_hours = 0.2
    env_params = {'max_time': 168.0} # 7 days is enough to show recovery after 72h shock
    
    env_inst = make_env_custom(sbml_dir, env_params, dt=dt_hours)
    dummy_env = DummyVecEnv([lambda: env_inst])
    env = VecNormalize.load(vec_path, dummy_env)
    env.training = False
    env.norm_reward = False
    
    if controller_type == 'rl':
        model = PPO.load(model_path, env=env)
    else:
        pi_controller = PIController(target_ph=7.0, kp=1.0, ki=0.1)

    base_env = env.venv.envs[0]
    obs = env.reset()
    
    data = []
    done = False
    shock_applied = False
    
    while not done:
        sim_state = base_env.simulator.state
        time_t = sim_state.time
        
        # Apply massive acid shock at 72 hours
        if time_t >= 72.0 and not shock_applied:
            # Drop pH to 4.5 by setting high proton concentration
            sim_state.metabolites['h_e'] = 10**(-4.5) * 1000.0
            shock_applied = True
            print(f"[{controller_type.upper()}] Massive acid shock applied at t={time_t}!")
            
        ph = sim_state.metabolites.get('h_e', 1e-7)
        ph_val = -np.log10(max(1e-12, ph) / 1000.0)
        biomass_or16 = sim_state.species['Actinoplanes_sp_OR16_lcp'].biomass if 'Actinoplanes_sp_OR16_lcp' in sim_state.species else 0
        biomass_ns21 = sim_state.species['Rhizobacter_gummiphilus_NS21'].biomass if 'Rhizobacter_gummiphilus_NS21' in sim_state.species else 0
        biomass_lp = sim_state.species['Lactobacillus_plantarum'].biomass if 'Lactobacillus_plantarum' in sim_state.species else 0
        total_pha = sum(s.pha_accumulated for s in sim_state.species.values())

        if controller_type == 'rl':
            action_pred, _ = model.predict(obs, deterministic=True)
            current_action = action_pred[0] if len(action_pred.shape) > 1 else action_pred
        else:
            current_action = pi_controller.get_action(ph_val)
                
        row = {
            'Time': time_t,
            'pH': ph_val,
            'Biomass_OR16': biomass_or16,
            'Biomass_NS21': biomass_ns21,
            'Biomass_LP': biomass_lp,
            'Total_PHA': total_pha
        }
        data.append(row)

        obs, rewards, dones, infos = env.step(np.array([current_action]))
        done = dones[0]
        
    df = pd.DataFrame(data)
    return df

def main():
    model_path = "outputs/checkpoints/ppo_godmode_v3_550000_steps.zip"
    vec_path = "outputs/checkpoints/ppo_godmode_v3_vecnormalize_550000_steps.pkl"
    sbml_dir = "models/sbml/final_consortium"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("Running RL Shock test...")
    df_rl = run_shock_simulation('rl', model_path, vec_path, sbml_dir)
    df_rl['Agent_Type'] = 'RL'

    print("Running PI Shock test...")
    df_pi = run_shock_simulation('pi', model_path, vec_path, sbml_dir)
    df_pi['Agent_Type'] = 'PI'

    df_combined = pd.concat([df_rl, df_pi], ignore_index=True)
    os.makedirs('paper_figures', exist_ok=True)
    df_combined.to_csv('paper_figures/fig7_shock_test.csv', index=False)
    print("Saved paper_figures/fig7_shock_test.csv")

if __name__ == "__main__":
    main()
