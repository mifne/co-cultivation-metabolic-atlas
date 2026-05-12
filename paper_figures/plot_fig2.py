import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# 出力品質の設定 (Publication Quality)
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

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)

# --- 2A: Learning Curve (Survival & Degradation) ---
ax_a = fig.add_subplot(gs[0, 0])
df_a = pd.read_csv('paper_figures/fig2A_learning_curve.csv')

# カラム名の特定 (サブエージェントの出力に依存しないよう柔軟に)
cols_a = df_a.columns
step_col = [c for c in cols_a if 'step' in c.lower()][0]
surv_col = [c for c in cols_a if 'survival' in c.lower()][0]

ax_a.plot(df_a[step_col], df_a[surv_col], color='#1f77b4', linewidth=2.5, alpha=0.9, label='Survival Hours')
ax_a.set_xlabel('Training Steps')
ax_a.set_ylabel('Survival Hours (h)', color='#1f77b4', fontweight='bold')
ax_a.tick_params(axis='y', labelcolor='#1f77b4')
ax_a.set_title('A. Learning Dynamics: Survival & Production', loc='left', fontweight='bold')

# ゴム分解量が存在する場合は第2軸にプロット
rub_cols = [c for c in cols_a if 'rubber' in c.lower() and 'deg' in c.lower()]
if rub_cols:
    ax_a2 = ax_a.twinx()
    ax_a2.plot(df_a[step_col], df_a[rub_cols[0]], color='#ff7f0e', linewidth=2.5, alpha=0.8, label='Rubber Degraded')
    ax_a2.set_ylabel('Total Rubber Degraded (g)', color='#ff7f0e', fontweight='bold')
    ax_a2.tick_params(axis='y', labelcolor='#ff7f0e')
    # 両方の凡例を表示
    lines, labels = ax_a.get_legend_handles_labels()
    lines2, labels2 = ax_a2.get_legend_handles_labels()
    ax_a.legend(lines + lines2, labels + labels2, loc='upper left')

# --- 2B: Tracking Time-course ---
ax_b = fig.add_subplot(gs[0, 1])
df_b = pd.read_csv('paper_figures/fig2B_timecourse.csv')
time_col = [c for c in df_b.columns if 'time' in c.lower()][0]

ax_b.plot(df_b[time_col], df_b['Biomass_OR16'], label='OR16 (Lcp Engine)', color='#2ca02c', linewidth=2.5)
ax_b.plot(df_b[time_col], df_b['Biomass_NS21'], label='NS21 (Rox/PHA Engine)', color='#d62728', linewidth=2.5)
ax_b.plot(df_b[time_col], df_b['Biomass_LP'], label='LP (Stabilizer)', color='#9467bd', linewidth=2.5)
ax_b.set_xlabel('Time (Hours)')
ax_b.set_ylabel('Biomass (g/L)')
ax_b.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.9)
ax_b.set_title('B. Population Tracking & Synergy', loc='left', fontweight='bold')

# pHを第2軸にプロット
ax_b2 = ax_b.twinx()
ax_b2.plot(df_b[time_col], df_b['pH'], label='pH Level', color='#8c564b', linestyle='--', linewidth=2.5, alpha=0.8)
ax_b2.set_ylabel('pH Level', color='#8c564b', fontweight='bold')
ax_b2.axhline(y=7.0, color='gray', linestyle=':', alpha=0.7, linewidth=1.5) # 理想値のライン
ax_b2.tick_params(axis='y', labelcolor='#8c564b')
ax_b2.set_ylim(4, 10)

# --- 2C: Value Heatmap ---
ax_c = fig.add_subplot(gs[1, 0])
df_cd = pd.read_csv('paper_figures/fig2CD_heatmap.csv')

# ピボットテーブルの作成
val_pivot = df_cd.pivot(index='NS21_biomass', columns='OR16_biomass', values='Value')
sns.heatmap(val_pivot, ax=ax_c, cmap='viridis', cbar_kws={'label': 'Predicted State Value ($V$)'})
ax_c.invert_yaxis() # 原点を左下に
ax_c.set_xlabel('OR16 Biomass Density (g/L)')
ax_c.set_ylabel('NS21 Biomass Density (g/L)')
ax_c.set_title('C. State Value Landscape', loc='left', fontweight='bold')

# 軸ラベルを適度に間引いて綺麗にする
xticks = ax_c.get_xticks()
yticks = ax_c.get_yticks()
step_x = max(1, len(xticks) // 5)
step_y = max(1, len(yticks) // 5)
ax_c.set_xticks(xticks[::step_x])
ax_c.set_yticks(yticks[::step_y])
ax_c.set_xticklabels([f"{val_pivot.columns[int(x)]:.1f}" for x in xticks[::step_x]])
ax_c.set_yticklabels([f"{val_pivot.index[int(y)]:.1f}" for y in yticks[::step_y]])

# --- 2D: Policy Heatmap (NS21 Specific Feed Action) ---
ax_d = fig.add_subplot(gs[1, 1])
# Action_1 は NS21 (sn_ns21) の供給量
act_pivot = df_cd.pivot(index='NS21_biomass', columns='OR16_biomass', values='Action_1')
sns.heatmap(act_pivot, ax=ax_d, cmap='coolwarm', cbar_kws={'label': 'Action: NS21 Specific Feed Rate'})
ax_d.invert_yaxis()
ax_d.set_xlabel('OR16 Biomass Density (g/L)')
ax_d.set_ylabel('NS21 Biomass Density (g/L)')
ax_d.set_title('D. Policy Decision Boundary (NS21 Nutrient)', loc='left', fontweight='bold')

ax_d.set_xticks(xticks[::step_x])
ax_d.set_yticks(yticks[::step_y])
ax_d.set_xticklabels([f"{act_pivot.columns[int(x)]:.1f}" for x in xticks[::step_x]])
ax_d.set_yticklabels([f"{act_pivot.index[int(y)]:.1f}" for y in yticks[::step_y]])

plt.tight_layout()
plt.savefig('paper_figures/Figure_2_Base_Control.png', bbox_inches='tight')
plt.savefig('paper_figures/Figure_2_Base_Control.pdf', bbox_inches='tight')
print("✅ Figure 2 generated: paper_figures/Figure_2_Base_Control.png")
