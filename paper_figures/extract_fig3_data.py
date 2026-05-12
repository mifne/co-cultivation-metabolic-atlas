import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from main import make_env

class PIController:
    def __init__(self, target_ph=7.0, kp=1.0, ki=0.1):
        self.target_ph = target_ph
        self.kp = kp
        self.ki = ki
        self.integral_error = 0.0
        
    def reset(self):
        self.integral_error = 0.0
        
    def get_action(self, current_ph):
        # We want to lower pH if it's > 7.0 by feeding LP.
        # LP produces lactic acid -> lower pH.
        # Error: if current_ph = 8.0, error = 1.0. We want positive action.
        error = current_ph - self.target_ph
        self.integral_error += error
        
        # Anti-windup
        self.integral_error = max(-10.0, min(10.0, self.integral_error))
        
        u = self.kp * error + self.ki * self.integral_error
        
        # action[2] is sn_lp, ranges 0 to 1
        sn_lp = max(0.0, min(1.0, u))
        
        # The baseline action
        # Let's keep others constant, maybe low.
        return np.array([0.05, 0.05, sn_lp, 0.05, 0.5])

def run_simulation(controller_type, interval_steps, model_path, vec_path, sbml_dir):
    env_params = {'max_time': 168.0}
    env_fn = make_env(sbml_dir, env_params)
    dummy_env = DummyVecEnv([env_fn])
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
    step_idx = 0
    current_action = np.zeros(5)
    
    while not done:
        sim_state = base_env.simulator.state
        time_t = sim_state.time
        ph = sim_state.metabolites.get('h_e', 1e-7)
        ph_val = -np.log10(max(1e-12, ph) / 1000.0)
        biomass_or16 = sim_state.species['Actinoplanes_sp_OR16_lcp'].biomass if 'Actinoplanes_sp_OR16_lcp' in sim_state.species else 0
        biomass_ns21 = sim_state.species['Rhizobacter_gummiphilus_NS21'].biomass if 'Rhizobacter_gummiphilus_NS21' in sim_state.species else 0
        biomass_lp = sim_state.species['Lactobacillus_plantarum'].biomass if 'Lactobacillus_plantarum' in sim_state.species else 0
        
        if step_idx % interval_steps == 0:
            if controller_type == 'rl':
                action_pred, _states = model.predict(obs, deterministic=True)
                current_action = action_pred[0] if len(action_pred.shape) > 1 else action_pred
            else:
                current_action = pi_controller.get_action(ph_val)
                
        row = {
            'Time': time_t,
            'pH': ph_val,
            'Biomass_OR16': biomass_or16,
            'Biomass_NS21': biomass_ns21,
            'Biomass_LP': biomass_lp,
            'Action_0': current_action[0],
            'Action_1': current_action[1],
            'Action_2': current_action[2],
            'Action_3': current_action[3],
            'Action_4': current_action[4],
        }
        data.append(row)

        obs, rewards, dones, infos = env.step(np.array([current_action]))
        done = dones[0]
        step_idx += 1
        
    df = pd.DataFrame(data)
    mse = np.mean((df['pH'] - 7.0)**2)
    return df, mse

def main():
    model_path = "outputs/final_v15_do_monitoring/ppo_consortium_2000000_steps.zip"
    vec_path = "outputs/final_v15_do_monitoring/ppo_consortium_vecnormalize_2000000_steps.pkl"
    sbml_dir = "models/sbml/final_consortium"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    intervals = [1, 3, 5, 10]
    
    # --- Task A ---
    print("Running Task A (MSE vs Interval)...")
    mse_results = []
    
    for iv in intervals:
        print(f"  Testing interval {iv} (RL)...")
        _, mse_rl = run_simulation('rl', iv, model_path, vec_path, sbml_dir)
        print(f"  Testing interval {iv} (PI)...")
        _, mse_pi = run_simulation('pi', iv, model_path, vec_path, sbml_dir)
        
        mse_results.append({
            'Interval': iv,
            'Interval_Minutes': iv * 12,
            'RL_MSE': mse_rl,
            'PI_MSE': mse_pi
        })
        
    df_mse = pd.DataFrame(mse_results)
    os.makedirs('paper_figures', exist_ok=True)
    df_mse.to_csv('paper_figures/fig3A_mse_sampling.csv', index=False)
    print("Saved paper_figures/fig3A_mse_sampling.csv")
    
    # --- Task B ---
    print("Running Task B (Time-course extraction)...")
    scenarios = [
        ('rl', 1),
        ('pi', 1),
        ('rl', 5),
        ('pi', 5)
    ]
    
    all_dfs = []
    for ctype, iv in scenarios:
        print(f"  Running {ctype.upper()} with interval {iv}...")
        df, _ = run_simulation(ctype, iv, model_path, vec_path, sbml_dir)
        df['Agent_Type'] = ctype.upper()
        df['Interval'] = iv
        all_dfs.append(df)
        
    df_combined = pd.concat(all_dfs, ignore_index=True)
    df_combined.to_csv('paper_figures/fig3B_E_timecourse.csv', index=False)
    print("Saved paper_figures/fig3B_E_timecourse.csv")

if __name__ == "__main__":
    main()
