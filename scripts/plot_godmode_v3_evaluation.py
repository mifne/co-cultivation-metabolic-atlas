import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    csv_path = Path("outputs/eval_results/best_model_timeseries_godmode.csv")
    output_path = Path("outputs/Figure_Evaluation_Result_v3.png")
    
    print(f"📊 Loading timeseries data from: {csv_path}")
    if not csv_path.exists():
        print(f"❌ Error: Timeseries CSV file not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # グラフの描画 (paper_figures 学術論文スタイル)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- パネルA: Population Dynamics & pH Control ---
    ax1 = axes[0]
    ax1.plot(df['Time'], df['Biomass_OR16'], label='OR16 (Rubber Degrader)', color='#2ca02c', linewidth=2.5)
    ax1.plot(df['Time'], df['Biomass_NS21'], label='NS21 (PHA Accumulator)', color='#d62728', linewidth=2.5)
    ax1.plot(df['Time'], df['Biomass_LP'], label='LP (Metabolic Buffer)', color='#9467bd', linewidth=2.5)
    
    ax1.set_xlabel('Time (Hours)', fontweight='bold')
    ax1.set_ylabel('Biomass (g/L)', fontweight='bold')
    ax1.set_title('A. Consortium Population Dynamics & pH Control', loc='left', fontweight='bold')
    
    ax1_ph = ax1.twinx()
    ax1_ph.plot(df['Time'], df['pH'], label='pH Level', color='#8c564b', linestyle='--', linewidth=2.5, alpha=0.8)
    ax1_ph.set_ylabel('pH Level', color='#8c564b', fontweight='bold')
    ax1_ph.axhline(y=7.0, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    ax1_ph.tick_params(axis='y', labelcolor='#8c564b')
    ax1_ph.set_ylim(4, 9)
    
    # 凡例を統合
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_ph.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, framealpha=0.9)
    
    # --- パネルB: Resource Conversion (Rubber to PHA) ---
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
    ax2_rub.set_ylim(-5, 105)
    
    # 凡例を統合
    lines_b, labels_b = ax2.get_legend_handles_labels()
    lines2_b, labels2_b = ax2_rub.get_legend_handles_labels()
    ax2.legend(lines_b + lines2_b, labels_b + labels2_b, loc='center right', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    
    # 出力ディレクトリ確保
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"🎉 High-resolution figure successfully saved to: {output_path}")

if __name__ == '__main__':
    main()
