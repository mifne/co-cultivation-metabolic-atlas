import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# 出力品質の設定 (Publication Quality)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], wspace=0.3, hspace=0.4)

df_tc = pd.read_csv('paper_figures/fig5_product_optimization_stochastic.csv')
time_col = [c for c in df_tc.columns if 'time' in c.lower()][0]

# --- 5A: Carbon Mass Balance / Ultimate Hand-off ---
ax_a = fig.add_subplot(gs[0, :])

# Plot Rubber Decreasing (Left Axis)
l1 = ax_a.plot(df_tc[time_col], df_tc['Rubber_Remaining'], color='black', linewidth=3, label='Rubber (Polymer)')
ax_a.set_xlabel('Time (Hours)')
ax_a.set_ylabel('Rubber Concentration (g/L)', color='black', fontweight='bold')
ax_a.tick_params(axis='y', labelcolor='black')
ax_a.set_ylim(0, 105)

# Plot Intermediates and Products (Right Axis)
ax_a2 = ax_a.twinx()
l2 = ax_a2.plot(df_tc[time_col], df_tc['C30_oligo_e'], color='#ff7f0e', linewidth=2.5, linestyle='--', label='C30 Oligomers')
l3 = ax_a2.plot(df_tc[time_col], df_tc['odtd_e'], color='#2ca02c', linewidth=2.5, linestyle='-.', label='ODTD')
l4 = ax_a2.plot(df_tc[time_col], df_tc['Total_PHA'], color='#d62728', linewidth=3, label='PHA (Product)')

ax_a2.set_ylabel('Concentration (mM / mmol)', color='#d62728', fontweight='bold')
ax_a2.tick_params(axis='y', labelcolor='#d62728')

# Combine legends
lines = l1 + l2 + l3 + l4
labels = [l.get_label() for l in lines]
ax_a.legend(lines, labels, loc='center left', frameon=True, fancybox=True, framealpha=0.9)
ax_a.set_title('A. Product Optimization: The Complete Metabolic Hand-off', loc='left', fontweight='bold')


# --- 5B: Degradation Rate vs Time ---
ax_b = fig.add_subplot(gs[1, 0])
# Calculate degradation rate (derivative of Rubber_Remaining)
rubber_diff = -np.diff(df_tc['Rubber_Remaining'])
dt = np.diff(df_tc[time_col])
deg_rate = np.concatenate(([0], rubber_diff / dt))

ax_b.plot(df_tc[time_col], deg_rate, color='black', linewidth=2)
ax_b.set_xlabel('Time (Hours)')
ax_b.set_ylabel('Degradation Rate (g/L/h)', fontweight='bold')
ax_b.set_title('B. Rubber Degradation Kinetics', loc='left', fontweight='bold')
ax_b.fill_between(df_tc[time_col], 0, deg_rate, color='black', alpha=0.1)


# --- 5C: Population Dynamics during Optimization ---
ax_c = fig.add_subplot(gs[1, 1])

ax_c.plot(df_tc[time_col], df_tc['Biomass_OR16'], label='OR16', color='#2ca02c', linewidth=2.5)
ax_c.plot(df_tc[time_col], df_tc['Biomass_NS21'], label='NS21', color='#d62728', linewidth=2.5)
ax_c.plot(df_tc[time_col], df_tc['Biomass_LP'], label='LP', color='#9467bd', linewidth=2.5)

ax_c.set_xlabel('Time (Hours)')
ax_c.set_ylabel('Biomass (g/L)')
ax_c.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.9)
ax_c.set_title('C. Consortium Population Dynamics', loc='left', fontweight='bold')

plt.tight_layout()
plt.savefig('paper_figures/Figure_5_Product_Optimization.png', bbox_inches='tight')
plt.savefig('paper_figures/Figure_5_Product_Optimization.pdf', bbox_inches='tight')
print("✅ Figure 5 generated: paper_figures/Figure_5_Product_Optimization.png")
