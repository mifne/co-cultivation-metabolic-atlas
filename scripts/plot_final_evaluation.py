import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dfba_simulator import dFBASimulator
from src.rl_environment import ConsortiumEnv
from src.utils import load_sbml_models, select_consortium_models, get_initial_params

def generate_evaluation_figure(model_path: str, sbml_dir: str, output_path: str):
    print("📊 評価データ収集開始...")
    
    # モデルの読み込み
    all_models = load_sbml_models(Path(sbml_dir))
    models = select_consortium_models(all_models)
    initial_biomass, initial_metabolites = get_initial_params(models)
    
    # シミュレーターのセットアップ
    sim = dFBASimulator(
        models=models,
        initial_biomass=initial_biomass,
        initial_metabolites=initial_metabolites,
        initial_rubber=100.0,
        volume=1.0,
        dt=0.2
    )
    
    # RL環境の作成
    env = ConsortiumEnv(simulator=sim, max_time=672.0)
    
    # ダミーベクター環境でラップしてVecNormalizeを適用
    venv = DummyVecEnv([lambda: env])
    
    path_obj = Path(model_path)
    stats_path = path_obj.parent / path_obj.name.replace("ppo_consortium", "ppo_consortium_vecnormalize").replace(".zip", ".pkl")
    
    if not stats_path.exists():
        print(f"❌ 正規化統計ファイルが見つかりません: {stats_path}")
        return
        
    venv = VecNormalize.load(str(stats_path), venv)
    venv.training = False
    venv.norm_reward = False
    
    # PPOエージェントの読み込み
    model = PPO.load(model_path, env=venv)
    
    # データを記録するリスト
    history = []
    
    obs = venv.reset()
    done = [False]
    
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        
        # 内部環境の生データにアクセス
        base_env = venv.venv.envs[0]
        state = base_env.simulator.state
        
        # pHの計算
        h_e = state.metabolites.get('h_e', 0.0001)
        ph = -np.log10(h_e / 1000.0) if h_e > 0 else 7.0
        
        history.append({
            'Time': state.time,
            'Biomass_OR16': next(s.biomass for n, s in state.species.items() if 'OR16' in n),
            'Biomass_NS21': next(s.biomass for n, s in state.species.items() if 'NS21' in n),
            'Biomass_LP': next(s.biomass for n, s in state.species.items() if 'Lactobacillus' in n),
            'pH': ph,
            'Rubber_Remaining': state.rubber_concentration,
            'Total_PHA': sum(s.pha_accumulated for s in state.species.values())
        })

    print("✅ データ収集完了。グラフを作成します...")
    df = pd.DataFrame(history)
    
    # --- グラフの描画 (paper_figures スタイル) ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- パネルA: Biomass & pH ---
    ax1 = axes[0]
    ax1.plot(df['Time'], df['Biomass_OR16'], label='OR16 (Rubber Degrader)', color='#2ca02c', linewidth=2.5)
    ax1.plot(df['Time'], df['Biomass_NS21'], label='NS21 (PHA Accumulator)', color='#d62728', linewidth=2.5)
    ax1.plot(df['Time'], df['Biomass_LP'], label='LP (Metabolic Buffer)', color='#9467bd', linewidth=2.5)
    
    ax1.set_xlabel('Time (Hours)', fontweight='bold')
    ax1.set_ylabel('Biomass (g/L)', fontweight='bold')
    ax1.set_title('A. Population Dynamics & pH Control', loc='left', fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.9)
    
    ax1_ph = ax1.twinx()
    ax1_ph.plot(df['Time'], df['pH'], label='pH Level', color='#8c564b', linestyle='--', linewidth=2.5, alpha=0.8)
    ax1_ph.set_ylabel('pH Level', color='#8c564b', fontweight='bold')
    ax1_ph.axhline(y=7.0, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    ax1_ph.tick_params(axis='y', labelcolor='#8c564b')
    ax1_ph.set_ylim(4, 10)
    
    # pHの凡例も追加
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_ph.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # --- パネルB: Rubber & PHA ---
    ax2 = axes[1]
    ax2.plot(df['Time'], df['Total_PHA'], label='Total PHA Accumulated', color='#1f77b4', linewidth=3.0)
    ax2.set_xlabel('Time (Hours)', fontweight='bold')
    ax2.set_ylabel('PHA Accumulated (mmol)', color='#1f77b4', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.set_title('B. Resource Conversion (Rubber to PHA)', loc='left', fontweight='bold')
    
    ax2_rub = ax2.twinx()
    ax2_rub.plot(df['Time'], df['Rubber_Remaining'], label='Rubber Remaining', color='#ff7f0e', linewidth=3.0, alpha=0.8)
    ax2_rub.set_ylabel('Rubber Concentration (g/L)', color='#ff7f0e', fontweight='bold')
    ax2_rub.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2_rub.set_ylim(0, 105)
    
    lines_b, labels_b = ax2.get_legend_handles_labels()
    lines2_b, labels2_b = ax2_rub.get_legend_handles_labels()
    ax2.legend(lines_b + lines2_b, labels_b + labels2_b, loc='center right')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"🎉 グラフを保存しました: {output_path}")

if __name__ == '__main__':
    model_path = 'outputs/checkpoints/ppo_consortium_520000_steps.zip'
    sbml_dir = 'models/sbml/final_consortium'
    output_path = 'outputs/Figure_Evaluation_Result.png'
    
    os.makedirs('outputs', exist_ok=True)
    generate_evaluation_figure(model_path, sbml_dir, output_path)
