import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.ppo_agent import ConsortiumPPOAgent
import cobra

def run_final_validation(model_path, sbml_dir):
    # 1. 環境とモデルのセットアップ
    all_models = {}
    for f in Path(sbml_dir).glob("*.xml"):
        model = cobra.io.read_sbml_model(str(f))
        all_models[f.stem] = model
    
    # メインの main.py と同じ初期化ロジック
    from main import get_initial_params, select_consortium_models
    models = select_consortium_models(all_models)
    initial_biomass, initial_metabolites = get_initial_params(models)
    
    sim = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=100.0,
        volume=1.0,
        dt=0.2,
        data_log_path="outputs/final_validation_telemetry.csv"
    )
    
    env = ConsortiumEnv(simulator=sim, max_time=168.0)
    agent = ConsortiumPPOAgent(env=env)
    agent.load(model_path)
    
    # 2. シミュレーション実行
    print("🚀 最終バリデーション実行中 (168時間)...")
    obs, _ = env.reset()
    done = False
    truncated = False
    
    history = []
    while not (done or truncated):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        # 現在の状態を保存
        state = env.simulator.state
        history.append({
            'time': state.time,
            'rubber': state.rubber_concentration,
            'glucose': state.metabolites.get('glc__D_e', 0),
            'fragments': state.metabolites.get('rubber_fragment_e', 0),
            'total_pha': sum(s.pha_accumulated for s in state.species.values()),
            'biomass_or16': state.species['Actinoplanes_sp_OR16_lcp'].biomass,
            'biomass_ns21': state.species['Rhizobacter_gummiphilus_NS21'].biomass,
            'biomass_lp': state.species['Lactobacillus_plantarum'].biomass,
            'ph': -np.log10(max(1e-12, state.metabolites.get('h_e', 0.0001)) / 1000.0)
        })
    
    df = pd.DataFrame(history)
    df.to_csv("outputs/final_validation_summary.csv", index=False)
    
    # 3. グラフ作成
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    # (A) Biomass & pH
    ax1 = axes[0]
    ax1.plot(df['time'], df['biomass_or16'], label='OR16 (qPCR)', color='blue', linewidth=2)
    ax1.plot(df['time'], df['biomass_ns21'], label='NS21 (qPCR)', color='green', linewidth=2)
    ax1.plot(df['time'], df['biomass_lp'], label='LP (qPCR)', color='red', linewidth=2)
    ax1.set_ylabel('Biomass Concentration (g/L)')
    ax1.legend(loc='upper left')
    
    ax1_ph = ax1.twinx()
    ax1_ph.plot(df['time'], df['ph'], '--', color='gray', alpha=0.5, label='pH')
    ax1_ph.set_ylabel('pH')
    ax1_ph.set_ylim(4, 9)
    ax1_ph.legend(loc='upper right')
    ax1.set_title("A. Species Population & pH Stability")
    
    # (B) Carbon Sources (Glucose & Rubber)
    ax2 = axes[1]
    ax2.plot(df['time'], df['glucose'], label='Glucose (LC/MS)', color='orange')
    ax2.set_ylabel('Glucose Concentration (mM)')
    ax2.legend(loc='upper left')
    
    ax2_r = ax2.twinx()
    # 変化を見やすくするために初期値からの差分を表示
    ax2_r.plot(df['time'], 100 - df['rubber'], label='Rubber Degraded (g/L)', color='black', linewidth=3)
    ax2_r.set_ylabel('Degradation Amount (g/L)')
    ax2_r.legend(loc='upper right')
    ax2.set_title("B. Carbon Source Dynamics & Rubber Degradation")
    
    # (C) Product (PHA) & Fragments
    ax3 = axes[2]
    ax3.plot(df['time'], df['total_pha'], label='Total PHA (LC/MS)', color='purple', linewidth=2)
    ax3.plot(df['time'], df['fragments'], label='Rubber Fragments (LC/MS)', color='cyan')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Concentration (mmol or mM)')
    ax3.legend()
    ax3.set_title("C. Product Accumulation (PHA)")
    
    plt.tight_layout()
    plt.savefig("outputs/ULTIMATE_SYSTEM_PROOF.png", dpi=200)
    print("✅ グラフを保存しました: outputs/ULTIMATE_SYSTEM_PROOF.png")
    
    final_deg = (100 - df['rubber'].iloc[-1]) / 100
    print(f"\n📊 最終評価結果:")
    print(f"   ゴム分解率: {final_deg:.8%}")
    print(f"   PHA生産量: {df['total_pha'].iloc[-1]:.4f} mmol")

if __name__ == "__main__":
    run_final_validation(
        "outputs/ultimate_consortium_v2/ppo_consortium_300000_steps.zip",
        "models/sbml/final_consortium"
    )
