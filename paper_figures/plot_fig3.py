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

fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1.2, 1, 1], wspace=0.3, hspace=0.4)

# --- 3A: MSE vs Sampling Interval ---
ax_a = fig.add_subplot(gs[:, 0])
df_mse = pd.read_csv('paper_figures/fig3A_mse_sampling.csv')

# バープロット用のデータ成形
x = np.arange(len(df_mse))
width = 0.35

ax_a.bar(x - width/2, df_mse['RL_MSE'], width, label='RL Controller', color='#1f77b4', alpha=0.9)
ax_a.bar(x + width/2, df_mse['PI_MSE'], width, label='PI Controller', color='#ff7f0e', alpha=0.9)

ax_a.set_ylabel('Average Sum Square Error (pH vs Target 7.0)')
ax_a.set_xlabel('Sampling Interval (minutes)')
ax_a.set_title('A. Robustness to Sampling Frequency', loc='left', fontweight='bold')
ax_a.set_xticks(x)
ax_a.set_xticklabels(df_mse['Interval_Minutes'])
ax_a.legend()

# y軸を対数スケールにすると差が見えやすい場合があるが、まずは線形
# ax_a.set_yscale('log')

# --- 3B-E 用のデータ準備 ---
df_tc = pd.read_csv('paper_figures/fig3B_E_timecourse.csv')

def plot_timecourse(ax, df_subset, title):
    # バイオマスのプロット
    l1 = ax.plot(df_subset['Time'], df_subset['Biomass_OR16'], label='OR16', color='#2ca02c', linewidth=2)
    l2 = ax.plot(df_subset['Time'], df_subset['Biomass_NS21'], label='NS21', color='#d62728', linewidth=2)
    l3 = ax.plot(df_subset['Time'], df_subset['Biomass_LP'], label='LP', color='#9467bd', linewidth=2)
    ax.set_xlabel('Time (Hours)')
    ax.set_ylabel('Biomass (g/L)')
    
    # pHのプロット (第2軸)
    ax2 = ax.twinx()
    l4 = ax2.plot(df_subset['Time'], df_subset['pH'], label='pH', color='#8c564b', linestyle='--', linewidth=2, alpha=0.8)
    ax2.axhline(y=7.0, color='gray', linestyle=':', linewidth=1.5)
    ax2.set_ylabel('pH Level', color='#8c564b')
    ax2.set_ylim(2, 12)
    ax2.tick_params(axis='y', labelcolor='#8c564b')
    
    ax.set_title(title, loc='left', fontweight='bold', fontsize=14)
    
    return l1 + l2 + l3 + l4

# --- 3B: RL 12 min ---
ax_b = fig.add_subplot(gs[0, 1])
df_rl_1 = df_tc[(df_tc['Agent_Type'] == 'RL') & (df_tc['Interval'] == 1)]
plot_timecourse(ax_b, df_rl_1, 'B. RL Control (12 min interval)')

# --- 3C: PI 12 min ---
ax_c = fig.add_subplot(gs[0, 2])
df_pi_1 = df_tc[(df_tc['Agent_Type'] == 'PI') & (df_tc['Interval'] == 1)]
lines = plot_timecourse(ax_c, df_pi_1, 'C. PI Control (12 min interval)')

# 凡例をCの図に配置
labels = [l.get_label() for l in lines]
ax_c.legend(lines, labels, loc='upper right', framealpha=0.9, fontsize=10)

# --- 3D: RL 60 min ---
ax_d = fig.add_subplot(gs[1, 1])
df_rl_5 = df_tc[(df_tc['Agent_Type'] == 'RL') & (df_tc['Interval'] == 5)]
plot_timecourse(ax_d, df_rl_5, 'D. RL Control (60 min interval)')

# --- 3E: PI 60 min ---
ax_e = fig.add_subplot(gs[1, 2])
df_pi_5 = df_tc[(df_tc['Agent_Type'] == 'PI') & (df_tc['Interval'] == 5)]
plot_timecourse(ax_e, df_pi_5, 'E. PI Control (60 min interval)')

plt.tight_layout()
plt.savefig('paper_figures/Figure_3_RL_vs_PI.png', bbox_inches='tight')
plt.savefig('paper_figures/Figure_3_RL_vs_PI.pdf', bbox_inches='tight')
print("✅ Figure 3 generated: paper_figures/Figure_3_RL_vs_PI.png")
