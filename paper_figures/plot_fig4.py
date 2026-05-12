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

# --- 4A: Online Learning (Survival Evolution) ---
ax_a = fig.add_subplot(gs[0, 0])

# Plotting the raw data lightly
ax_a.scatter(df['step'], df['survival_hours'], color='#1f77b4', alpha=0.2, s=10, label='Episode Survival')

# Plotting the rolling average prominently
ax_a.plot(df['step'], df['survival_hours_rolling'], color='#d62728', linewidth=3, label='10-Episode Rolling Avg')

# Target line (168 hours)
ax_a.axhline(y=168.0, color='gray', linestyle='--', linewidth=2, alpha=0.8, label='Target (168h)')

ax_a.set_xlabel('Training Steps')
ax_a.set_ylabel('Survival Hours')
ax_a.set_title('A. Evolution of Consortium Stability', loc='left', fontweight='bold')
ax_a.legend(loc='lower right', frameon=True, fancybox=True, framealpha=0.9)
ax_a.set_ylim(0, 180)

# --- 4B: Biomass Distribution at Final Stage (Pie Chart) ---
# To show the composition of the stable consortium at the end of training
ax_b = fig.add_subplot(gs[0, 1])

# Load timecourse data to get final biomass ratios from the best run (Fig2B data)
df_tc = pd.read_csv('paper_figures/fig2B_timecourse.csv')
final_state = df_tc.iloc[-1]

labels = ['OR16\n(Lcp Engine)', 'NS21\n(Rox/PHA Engine)', 'LP\n(Stabilizer)']
sizes = [final_state['Biomass_OR16'], final_state['Biomass_NS21'], final_state['Biomass_LP']]
colors = ['#2ca02c', '#d62728', '#9467bd']
explode = (0.05, 0.05, 0)  # explode the degradation engines slightly

ax_b.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=False, startangle=90, textprops={'weight': 'bold'})
ax_b.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax_b.set_title('B. Final Consortium Composition', loc='left', fontweight='bold', pad=20)


plt.tight_layout()
plt.savefig('paper_figures/Figure_4_Survival_Stability.png', bbox_inches='tight')
plt.savefig('paper_figures/Figure_4_Survival_Stability.pdf', bbox_inches='tight')
print("✅ Figure 4 generated: paper_figures/Figure_4_Survival_Stability.png")
