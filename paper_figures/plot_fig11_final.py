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
    'legend.fontsize': 13,
    'figure.titlesize': 22,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

def plot_figure_11_final():
    print("Plotting Final Figure 11 (Refined Models & Deprivation Experiment)...")
    try:
        df = pd.read_csv('paper_figures/fig11_final_data.csv')
    except Exception as e:
        print(f"Error loading fig11_final_data.csv: {e}")
        return

    df_god = df[df['Agent'] == 'God-Mode']
    df_pomdp = df[df['Agent'] == 'Real-World (POMDP)']
    df_deprived = df[df['Agent'] == 'God-Mode (Forced POMDP)']

    fig = plt.figure(figsize=(16, 7.5))
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)

    # Panel A: Total PHA Accumulation
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(df_god['time'], df_god['pha'], color='#2c3e50', linewidth=3.5, label='Omniscient AI (God-Mode)')
    ax_a.plot(df_pomdp['time'], df_pomdp['pha'], color='#e67e22', linewidth=3, linestyle='--', label='Real-World AI (POMDP)')
    ax_a.plot(df_deprived['time'], df_deprived['pha'], color='#9b59b6', linewidth=2.5, linestyle='-.', label='Deprived God-Mode (Forced POMDP)')
    
    ax_a.set_xlabel('Time (Hours)', fontweight='bold')
    ax_a.set_ylabel('Total PHA Accumulated (mmol)', fontweight='bold')
    ax_a.set_title('A. Product Yield Comparison', loc='left', fontweight='bold')
    ax_a.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='gray')
    ax_a.set_xlim(0, 168)

    # Panel B: pH Stability Control
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(df_god['time'], df_god['pH'], color='#2c3e50', linewidth=3, alpha=0.9, label='Omniscient AI')
    ax_b.plot(df_pomdp['time'], df_pomdp['pH'], color='#e67e22', linewidth=2.5, alpha=0.9, linestyle='--', label='POMDP')
    ax_b.plot(df_deprived['time'], df_deprived['pH'], color='#9b59b6', linewidth=2, alpha=0.9, linestyle='-.', label='Deprived God-Mode')
    
    ax_b.axhline(y=7.0, color='gray', linestyle=':', linewidth=2)
    ax_b.set_ylim(2.5, 8.5) # 全体の崩壊を完全に捉えるためにレンジを拡大
    ax_b.set_xlabel('Time (Hours)', fontweight='bold')
    ax_b.set_ylabel('pH Level', fontweight='bold')
    ax_b.set_title('B. Metabolic Stability Control', loc='left', fontweight='bold')
    ax_b.set_xlim(0, 168)
    ax_b.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='gray')
    
    ax_b.annotate('POMDP: pH collapse to 2.99\ndue to runaway lactate fermentation',
                  xy=(60, 4.0), xytext=(85, 4.5),
                  arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                  fontsize=12, style='italic', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                  
    ax_b.annotate('Deprived God-Mode:\nZero-growth conservative state',
                  xy=(110, 7.35), xytext=(45, 6.0),
                  arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                  fontsize=12, style='italic', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    output_pdf = 'paper_figures/Figure_11_Final_Comparison.pdf'
    output_png = 'paper_figures/Figure_11_Final_Comparison.png'
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.savefig(output_png, bbox_inches='tight')
    print(f"Saved {output_png}")

if __name__ == "__main__":
    plot_figure_11_final()
