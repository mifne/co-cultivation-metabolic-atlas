import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.ppo_agent import ConsortiumPPOAgent
import cobra
import os

def run_long_term_validation(model_path, sbml_dir, total_hours=672.0):
    # 1. 環境とモデルのセットアップ
    all_models = {}
    for f in Path(sbml_dir).glob("*.xml"):
        model = cobra.io.read_sbml_model(str(f))
        all_models[f.stem] = model
    
    from main import get_initial_params, select_consortium_models
    models = select_consortium_models(all_models)
    initial_biomass, initial_metabolites = get_initial_params(models)
    
    # 学習時と同じ dt=0.2 を使用して時間解像度を合わせる
    sim = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=100.0,
        volume=1.0,
        dt=0.2, 
        data_log_path="outputs/ultimate_long_term_telemetry_v3.csv"
    )
    
    # 訓練時と同じ max_time=672.0 に設定
    env = ConsortiumEnv(simulator=sim, max_time=672.0)
    agent = ConsortiumPPOAgent(env=env)
    
    if not os.path.exists(model_path):
        print(f"❌ モデルが見つかりません: {model_path}")
        return
        
    agent.load(model_path)
    
    # 2. シミュレーション実行 (学習なしの純粋な制御)
    print(f"🚀 長期バリデーション実行中 ({total_hours}時間)...")
    obs, _ = env.reset()
    
    history = []
    current_time = 0.0
    
    while current_time < total_hours:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        state = env.simulator.state
        current_time = state.time
        
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
        
        if int(current_time) % 48 == 0 and (current_time - int(current_time)) < 0.2:
            print(f"  🕒 {current_time:.0f}h 経過 (pH: {history[-1]['ph']:.2f}, Rubber: {history[-1]['rubber']:.4f})")
            
        if terminated:
            print(f"💀 {current_time:.1f}h で菌が死滅しました。")
            break

    df = pd.DataFrame(history)
    df.to_csv("outputs/ultimate_long_term_summary_v3.csv", index=False)
    
    # 3. グラフ作成
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    # (A) Biomass & pH
    ax1 = axes[0]
    ax1.plot(df['time'], df['biomass_or16'], label='OR16 (LCP)', color='blue')
    ax1.plot(df['time'], df['biomass_ns21'], label='NS21 (PHA)', color='green')
    ax1.plot(df['time'], df['biomass_lp'], label='LP (Acid)', color='red')
    ax1.set_ylabel('Biomass (g/L)')
    ax1.legend(loc='upper left')
    
    ax1_ph = ax1.twinx()
    ax1_ph.plot(df['time'], df['ph'], '--', color='orange', alpha=0.6, label='pH')
    ax1_ph.axhline(7.0, color='black', linestyle=':', alpha=0.5)
    ax1_ph.axhline(3.0, color='red', linestyle='--', alpha=0.3, label='Lethal pH')
    ax1_ph.set_ylabel('pH')
    ax1_ph.set_ylim(2, 9)
    ax1_ph.legend(loc='upper right')
    ax1.set_title(f"A. Long-term Population & pH Dynamics (V3 FBA-Policy)")
    
    # (B) Rubber Degradation
    ax2 = axes[1]
    degradation_g_l = 100.0 - df['rubber']
    ax2.plot(df['time'], degradation_g_l, color='black', linewidth=2.5, label='Rubber Degraded')
    ax2.set_ylabel('Cumulative Degradation (g/L)')
    ax2.set_xlabel('Time (hours)')
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='upper left')
    ax2.set_title("B. Rubber Degradation Progress (Mass Balance Validated)")
    
    # (C) Product Accumulation
    ax3 = axes[2]
    ax3.plot(df['time'], df['total_pha'], label='Total PHA', color='purple', linewidth=2)
    ax3.plot(df['time'], df['fragments'], label='Rubber Fragments', color='cyan', alpha=0.6)
    ax3.set_ylabel('Concentration (mmol or mM)')
    ax3.set_xlabel('Time (hours)')
    ax3.legend()
    ax3.set_title("C. Bio-Product (PHA) & Metabolic Intermediates")
    
    plt.tight_layout()
    plt.savefig("outputs/ULTIMATE_V3_FBA_PROOF.png", dpi=200)
    print("✅ グラフを保存しました: outputs/ULTIMATE_V3_FBA_PROOF.png")
    
    final_deg_pct = (100 - df['rubber'].iloc[-1]) / 100 * 100
    print(f"\n📊 4週間(672h)長期バリデーション結果 (V3):")
    print(f"   最終ゴム分解率: {final_deg_pct:.6f} %")
    print(f"   最終PHA生産量: {df['total_pha'].iloc[-1]:.6f} mmol")
    print(f"   最終pH: {df['ph'].iloc[-1]:.2f}")

if __name__ == "__main__":
    # 最新の学習モデルを指定
    model_file = "outputs/ultimate_consortium_v3_fba_single/ppo_consortium_model.zip"
    if not os.path.exists(model_file):
        # 学習途中の最新チェックポイントを探す
        checkpoints = list(Path("outputs/ultimate_consortium_v3_fba_single").glob("ppo_consortium_*.zip"))
        if checkpoints:
            model_file = str(sorted(checkpoints)[-1])
            print(f"ℹ️  学習途中モデルを使用します: {model_file}")

    run_long_term_validation(
        model_file,
        "models/sbml/final_consortium",
        total_hours=672.0
    )
