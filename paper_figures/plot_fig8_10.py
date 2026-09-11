import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Publication Quality Settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 22,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

def plot_figure_8():
    print("Plotting Figure 8 (XAI Feature Importance)...")
    try:
        df = pd.read_csv('paper_figures/fig8_feature_importance.csv')
    except Exception as e:
        print(f"Error loading fig8_feature_importance.csv: {e}")
        return

    # Pivot table for heatmap
    pivot_df = df.pivot(index='Action', columns='Feature', values='Importance')
    
    # Sort columns by mean importance to show most important features first
    sorted_cols = pivot_df.mean().sort_values(ascending=False).index
    pivot_df = pivot_df[sorted_cols]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.2f', 
                cbar_kws={'label': 'Random Forest Feature Importance'}, ax=ax, linewidths=0.5)
    
    ax.set_title('Explainable AI: Sensor Importance per Control Action', fontweight='bold', pad=20)
    ax.set_xlabel('Sensor Input (Observations)', fontweight='bold', labelpad=15)
    ax.set_ylabel('RL Agent Action', fontweight='bold', labelpad=15)
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('paper_figures/Figure_8_XAI_Heatmap.pdf', bbox_inches='tight')
    plt.savefig('paper_figures/Figure_8_XAI_Heatmap.png', bbox_inches='tight')
    print("Saved Figure_8_XAI_Heatmap")

def plot_figure_10():
    print("Plotting Figure 10 (Metabolic Pacing)...")
    try:
        df = pd.read_csv('paper_figures/fig10_pacing_data.csv')
    except Exception as e:
        print(f"Error loading fig10_pacing_data.csv: {e}")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # X-axis is time. We plot C30 concentration vs OR16 Feed Action.
    ax1.plot(df['Time'], df['C30_oligo_e'], color='#d62728', linewidth=3, label='C30 Oligomer Conc.')
    ax1.set_xlabel('Time (Hours)', fontweight='bold')
    ax1.set_ylabel('C30 Oligomer Concentration (mM)', color='#d62728', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#d62728')

    ax2 = ax1.twinx()
    ax2.fill_between(df['Time'], 0, df['Feed_OR16'], color='#2ca02c', alpha=0.3, label='OR16 Feed (Action 0)')
    ax2.plot(df['Time'], df['Feed_OR16'], color='#2ca02c', linewidth=2)
    ax2.set_ylabel('RL Feed Action (OR16)', color='#2ca02c', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    
    ax1.set_title('AI Pacing Strategy: Toxicty Avoidance', loc='left', fontweight='bold')
    
    # Add annotation pointing out the negative correlation
    ax1.annotate('Agent halts OR16 feed\nwhen C30 accumulates', 
                 xy=(50, df['C30_oligo_e'].max()*0.8), xytext=(80, df['C30_oligo_e'].max()*0.9),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, style='italic')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.tight_layout()
    plt.savefig('paper_figures/Figure_10_Pacing_Strategy.pdf', bbox_inches='tight')
    plt.savefig('paper_figures/Figure_10_Pacing_Strategy.png', bbox_inches='tight')
    print("Saved Figure_10_Pacing_Strategy")

if __name__ == "__main__":
    plot_figure_8()
    plot_figure_10()
