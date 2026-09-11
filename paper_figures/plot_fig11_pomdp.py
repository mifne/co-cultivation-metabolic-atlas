import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def plot_figure_11():
    print("Plotting Figure 11 (God-Mode vs Real-World POMDP)...")
    try:
        df = pd.read_csv('paper_figures/fig11_pomdp_comparison.csv')
    except Exception as e:
        print(f"Error loading fig11_pomdp_comparison.csv: {e}")
        return

    df_god = df[df['Agent_Type'] == 'God-Mode']
    df_pomdp = df[df['Agent_Type'] == 'Real-World (POMDP)']

    if df_god.empty or df_pomdp.empty:
        print("Waiting for both agents data...")
        return

    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)

    # Panel A: Total PHA Accumulation
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(df_god['Time'], df_god['Total_PHA'], color='#1f77b4', linewidth=3, label='Omniscient (God-Mode) AI')
    ax_a.plot(df_pomdp['Time'], df_pomdp['Total_PHA'], color='#ff7f0e', linewidth=3, linestyle='--', label='Real-World (POMDP) AI')
    
    ax_a.set_xlabel('Time (Hours)', fontweight='bold')
    ax_a.set_ylabel('Total PHA Accumulated (mmol)', fontweight='bold')
    ax_a.set_title('A. Product Yield Comparison', loc='left', fontweight='bold')
    ax_a.legend(loc='lower right')

    # Panel B: pH Stability Control
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(df_god['Time'], df_god['pH'], color='#1f77b4', linewidth=2, alpha=0.8, label='God-Mode AI')
    ax_b.plot(df_pomdp['Time'], df_pomdp['pH'], color='#ff7f0e', linewidth=2, alpha=0.8, label='POMDP AI')
    
    ax_b.axhline(y=7.0, color='gray', linestyle=':', linewidth=2)
    ax_b.set_ylim(4, 10)
    ax_b.set_xlabel('Time (Hours)', fontweight='bold')
    ax_b.set_ylabel('pH Level', fontweight='bold')
    ax_b.set_title('B. pH Stability Control', loc='left', fontweight='bold')
    
    ax_b.annotate('POMDP Agent successfully infers\ninternal states from pH and DO alone',
                  xy=(df_pomdp['Time'].max()*0.2, 8.5),
                  fontsize=12, style='italic', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    plt.savefig('paper_figures/Figure_11_POMDP_Comparison.pdf', bbox_inches='tight')
    plt.savefig('paper_figures/Figure_11_POMDP_Comparison.png', bbox_inches='tight')
    print("Saved Figure_11_POMDP_Comparison")

if __name__ == "__main__":
    plot_figure_11()
