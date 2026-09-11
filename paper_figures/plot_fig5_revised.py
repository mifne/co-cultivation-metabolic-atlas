import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 出力品質の設定 (Publication Quality)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], wspace=0.3, hspace=0.4)

df_det = pd.read_csv('paper_figures/fig5_deterministic.csv')
df_hyb = pd.read_csv('paper_figures/fig5_hybrid.csv')

def plot_panel(ax, df, title):
    # Convert to Carbon-mmol (C-mM) for mass balance
    df['Rubber_CmM'] = df['Rubber_Remaining'] * 73.4
    df['C30_oligo_CmM'] = df['C30_oligo_e'] * 30.0
    df['odtd_CmM'] = df['odtd_e'] * 15.0
    df['Total_PHA_CmM'] = df['Total_PHA'] * 4.0

    # Plot All on Single Axis
    l1 = ax.plot(df['Time'], df['Rubber_CmM'], color='black', linewidth=3, label='Rubber (Polymer)')
    l2 = ax.plot(df['Time'], df['C30_oligo_CmM'], color='#ff7f0e', linewidth=2.5, linestyle='--', label='C30 Oligomers')
    l3 = ax.plot(df['Time'], df['odtd_CmM'], color='#2ca02c', linewidth=2.5, linestyle='-.', label='ODTD')
    l4 = ax.plot(df['Time'], df['Total_PHA_CmM'], color='#d62728', linewidth=3, label='PHA (Product)')

    ax.set_xlabel('Time (Hours)')
    ax.set_ylabel('Carbon Concentration (C-mM)', fontweight='bold')
    ax.set_ylim(0, 8000)

    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    # 凡例をグラフの外（下部）に配置して線との重なりを完全に防ぐ
    ax.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=2, frameon=True, fancybox=True, framealpha=0.9)
    ax.set_title(title, loc='left', fontweight='bold')

# 5A: Deterministic Run
ax_a = fig.add_subplot(gs[0, 0])
plot_panel(ax_a, df_det, 'A. Learned Policy (Deterministic): Maximizing Degradation')

# 5B: Hybrid Run
ax_b = fig.add_subplot(gs[0, 1])
plot_panel(ax_b, df_hyb, 'B. Biological Potential (Hybrid): Forced Starvation at t=336h')

# 5C: Action (YE Feed) Comparison
ax_c = fig.add_subplot(gs[1, :])
ax_c.plot(df_det['Time'], df_det['YE_Feed'], color='#1f77b4', linewidth=2.5, label='Learned Policy (Continuous N-Feed)')
ax_c.plot(df_hyb['Time'], df_hyb['YE_Feed'], color='#d62728', linewidth=2.5, linestyle='--', label='Hybrid Policy (Forced Starvation)')
ax_c.set_xlabel('Time (Hours)')
ax_c.set_ylabel('Nitrogen Feed Action (0-1)')
# こちらの凡例も外に配置
ax_c.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=True)
ax_c.set_title('C. Nitrogen Feeding Strategy Comparison', loc='left', fontweight='bold')
ax_c.axvline(x=336.0, color='gray', linestyle=':', linewidth=2)
ax_c.text(340, 0.5, 'Starvation Triggered', color='gray', fontsize=14, rotation=90, verticalalignment='center')

plt.tight_layout()
plt.savefig('paper_figures/Figure_5_Revised.png', bbox_inches='tight')
plt.savefig('paper_figures/Figure_5_Revised.pdf', bbox_inches='tight')
print("✅ Figure 5 (Revised) generated.")
