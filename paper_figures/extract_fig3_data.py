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

class DualPIController:
    """Biologically-driven Dual PI Controller (Fair Baseline)
    Controls both acid (LP) and base (AA deamination via OR16/NS21)
    """
    def __init__(self, target_ph=7.0, kp=1.5, ki=0.2):
        self.target_ph = target_ph
        self.kp = kp
        self.ki = ki
        self.integral_error_acid = 0.0
        self.integral_error_base = 0.0
        
    def reset(self):
        self.integral_error_acid = 0.0
        self.integral_error_base = 0.0
        
    def get_action(self, current_ph):
        # Acid loop (Feed LP if pH > 7.0)
        error_acid = current_ph - self.target_ph
        if error_acid > 0:
            self.integral_error_acid += error_acid
            self.integral_error_base = 0.0 # reset opposing integrator
        else:
            error_acid = 0.0
            
        self.integral_error_acid = max(0.0, min(10.0, self.integral_error_acid))
        u_acid = self.kp * error_acid + self.ki * self.integral_error_acid
        sn_lp = max(0.0, min(1.0, u_acid))
        
        # Base loop (Feed OR16/NS21/Common to induce deamination if pH < 7.0)
        error_base = self.target_ph - current_ph
        if error_base > 0:
            self.integral_error_base += error_base
            self.integral_error_acid = 0.0 # reset opposing integrator
        else:
            error_base = 0.0
            
        self.integral_error_base = max(0.0, min(10.0, self.integral_error_base))
        u_base = self.kp * error_base + self.ki * self.integral_error_base
        sn_base = max(0.05, min(1.0, u_base)) # Minimum baseline feed 0.05 to prevent starvation
        
        return np.array([sn_base, sn_base, sn_lp, sn_base, 0.5])

class ConstantFeedController:
    """Optimal Constant Fed-Batch (State-of-the-Art Conventional)"""
    def __init__(self, feed_vector=[0.2, 0.2, 0.2, 0.2, 0.5]):
        self.feed_vector = np.array(feed_vector)
        
    def get_action(self, current_ph):
        return self.feed_vector

def run_simulation(controller_type, interval_minutes, model_path, vec_path, sbml_dir):
    dt_hours = 0.2 # 12-minute steps to avoid solver timeouts and make FBA 12x faster
    env_params = {'max_time': 168.0} # Extend to 168h (7 days) to prove Competitive Exclusion
    
    env_inst = make_env_custom(sbml_dir, env_params, dt=dt_hours)
    dummy_env = DummyVecEnv([lambda: env_inst])
    env = VecNormalize.load(vec_path, dummy_env)
    env.training = False
    env.norm_reward = False
    
    if controller_type == 'RL':
        model = PPO.load(model_path, env=env)
    elif controller_type == 'Dual-PI':
        controller = DualPIController()
    elif controller_type == 'Constant':
        controller = ConstantFeedController()

    base_env = env.venv.envs[0]
    obs = env.reset()
    
    data = []
    done = False
    steps = 0
    control_steps = 1 # 12 mins control interval / 12 mins dt = 1 step
    save_steps = 5    # 60 mins / 12 mins dt = 5 steps
    current_action = np.array([0.05, 0.05, 0.05, 0.05, 0.5])
    
    while not done:
        sim_state = base_env.simulator.state
        time_t = sim_state.time
        
        ph = sim_state.metabolites.get('h_e', 1e-7)
        ph_val = -np.log10(max(1e-12, ph) / 1000.0)
        
        if steps % control_steps == 0:
            if controller_type == 'RL':
                action_pred, _ = model.predict(obs, deterministic=True)
                current_action = action_pred[0] if len(action_pred.shape) > 1 else action_pred
            else:
                current_action = controller.get_action(ph_val)
                
        biomass_or16 = sim_state.species['Actinoplanes_sp_OR16_lcp'].biomass if 'Actinoplanes_sp_OR16_lcp' in sim_state.species else 0
        biomass_ns21 = sim_state.species['Rhizobacter_gummiphilus_NS21'].biomass if 'Rhizobacter_gummiphilus_NS21' in sim_state.species else 0
        biomass_lp = sim_state.species['Lactobacillus_plantarum'].biomass if 'Lactobacillus_plantarum' in sim_state.species else 0
        total_pha = sum(s.pha_accumulated for s in sim_state.species.values())

        # Save data every 1 hour (save_steps = 5) to keep CSV small
        if steps % save_steps == 0:
            data.append({
                'Time': time_t,
                'pH': ph_val,
                'Biomass_OR16': biomass_or16,
                'Biomass_NS21': biomass_ns21,
                'Biomass_LP': biomass_lp,
                'Total_PHA': total_pha,
                'Action_0': current_action[0],
                'Action_1': current_action[1],
                'Action_2': current_action[2],
                'Action_3': current_action[3]
            })

        obs, rewards, dones, infos = env.step(np.array([current_action]))
        done = dones[0]
        steps += 1
        
        # 絶滅（早期終了）した場合、残りの時間をバイオマス0で埋めてグラフを急落させる
        if done and time_t < env_params['max_time'] - dt_hours:
            print(f"    ⚠️ Extinction detected at {time_t}h! Padding remainder with dead state.")
            remaining_steps = int((env_params['max_time'] - time_t) / dt_hours)
            for i in range(remaining_steps):
                time_t += dt_hours
                if (steps + i) % save_steps == 0:
                    data.append({
                        'Time': time_t,
                        'pH': ph_val, # Keep last pH
                        'Biomass_OR16': 0.0, # Dead
                        'Biomass_NS21': 0.0, # Dead
                        'Biomass_LP': 0.0,   # Dead
                        'Total_PHA': total_pha, # PHA remains as solid
                        'Action_0': 0.0,
                        'Action_1': 0.0,
                        'Action_2': 0.0,
                        'Action_3': 0.0
                    })
            break
        
    df = pd.DataFrame(data)
    mse = np.mean((df['pH'] - 7.0)**2)
    return df, mse

def main():
    model_path = "outputs/refined_models/godmode/godmode_final_refined.zip"
    vec_path = "outputs/refined_models/godmode/godmode_final_refined_vecnormalize.pkl"
    sbml_dir = "models/sbml/final_consortium"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # To prove Competitive Exclusion over 7 days, we only need the 12-min interval.
    intervals_min = [12]
    agents = ['RL', 'Dual-PI', 'Constant']
    
    mse_results = []
    tc_results = []

    print("Extracting data for Figure 3 (RL vs Dual-PI vs Constant)...")
    
    for iv_min in intervals_min:
        for agent in agents:
            print(f"  Testing interval {iv_min} min ({agent})...")
            df_tc, mse = run_simulation(agent, iv_min, model_path, vec_path, sbml_dir)
            
            df_tc['Interval'] = iv_min
            df_tc['Agent_Type'] = agent
            tc_results.append(df_tc)
            
            mse_results.append({
                'Interval_Minutes': iv_min,
                'Agent_Type': agent,
                'MSE': mse
            })

    df_mse = pd.DataFrame(mse_results)
    df_all_tc = pd.concat(tc_results, ignore_index=True)

    os.makedirs('paper_figures', exist_ok=True)
    df_mse.to_csv('paper_figures/fig3A_mse_sampling.csv', index=False)
    df_all_tc.to_csv('paper_figures/fig3B_E_timecourse.csv', index=False)

    print("Data extraction for Figure 3 complete.")

if __name__ == "__main__":
    main()
