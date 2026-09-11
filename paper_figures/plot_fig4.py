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

fig = plt.figure(figsize=(14, 8))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1], wspace=0.3)

df = pd.read_csv('paper_figures/fig4_survival_evolution.csv')

# --- 4A: Online Learning (Survival & Degradation) ---
ax_a = fig.add_subplot(gs[0, 0])

# Plotting the Survival Hours
ax_a.plot(df['step'], df['survival_hours_rolling'], color='#1f77b4', linewidth=3, label='Survival Hours (10-Ep Avg)')
ax_a.set_xlabel('Training Steps')
ax_a.set_ylabel('Survival Hours', color='#1f77b4', fontweight='bold')
ax_a.tick_params(axis='y', labelcolor='#1f77b4')
ax_a.set_ylim(0, max(200, df['survival_hours_rolling'].max() * 1.2))

# Plotting Total Rubber Degraded on secondary axis
ax_a2 = ax_a.twinx()
ax_a2.plot(df['step'], df['total_rubber_degraded_rolling'], color='#d62728', linewidth=3, linestyle='--', label='Rubber Degraded (10-Ep Avg)')
ax_a2.set_ylabel('Total Rubber Degraded (g/L)', color='#d62728', fontweight='bold')
ax_a2.tick_params(axis='y', labelcolor='#d62728')
ax_a2.set_ylim(0, 110)

ax_a.set_title('A. Evolution of Survival & Degradation Efficiency', loc='left', fontweight='bold')

# Combine legends
lines_a = ax_a.get_lines() + ax_a2.get_lines()
labels_a = [l.get_label() for l in lines_a]
ax_a.legend(lines_a, labels_a, loc='lower right', frameon=True, fancybox=True, framealpha=0.9)

# Adding an annotation about starvation
ax_a.annotate('Shorter survival indicates faster\ncomplete degradation (starvation)',
              xy=(df['step'].max() * 0.8, 110), xycoords='data',
              xytext=(df['step'].max() * 0.5, 50), textcoords='data',
              arrowprops=dict(facecolor='gray', shrink=0.05, alpha=0.7),
              fontsize=12, color='gray', style='italic')


# --- 4B: Biomass Distribution at Final Stage (Stacked Area Chart) ---
# To show the composition of the stable consortium over time
ax_b = fig.add_subplot(gs[0, 1])

# Load timecourse data to get biomass ratios over time
df_tc = pd.read_csv('paper_figures/fig2B_timecourse.csv')

total_biomass = df_tc['Biomass_OR16'] + df_tc['Biomass_NS21'] + df_tc['Biomass_LP']
or16_ratio = df_tc['Biomass_OR16'] / total_biomass * 100.0
ns21_ratio = df_tc['Biomass_NS21'] / total_biomass * 100.0
lp_ratio = df_tc['Biomass_LP'] / total_biomass * 100.0

ax_b.stackplot(df_tc['Time'], or16_ratio, ns21_ratio, lp_ratio,
               labels=['OR16 (Lcp)', 'NS21 (Rox/PHA)', 'LP (Stabilizer)'],
               colors=['#2ca02c', '#d62728', '#9467bd'], alpha=0.8)

ax_b.set_xlabel('Time (Hours)')
ax_b.set_ylabel('Relative Abundance (%)', fontweight='bold')
ax_b.set_ylim(0, 100)
ax_b.set_xlim(0, df_tc['Time'].max())
ax_b.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
ax_b.set_title('B. Consortium Population Dynamics', loc='left', fontweight='bold')


plt.tight_layout()
plt.savefig('paper_figures/Figure_4_Survival_Stability.png', bbox_inches='tight')
plt.savefig('paper_figures/Figure_4_Survival_Stability.pdf', bbox_inches='tight')
print("✅ Figure 4 generated: paper_figures/Figure_4_Survival_Stability.png")
