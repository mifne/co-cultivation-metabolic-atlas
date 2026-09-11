import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# 出力品質の設定 (Publication Quality)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

# --- 3A: Comprehensive Performance Comparison (PHA & Rubber) ---
ax_a = fig.add_subplot(gs[0, :])
try:
    df_tc_all = pd.read_csv('paper_figures/fig3B_E_timecourse.csv')
    
    summary = []
    for agent in ['RL', 'Dual-PI', 'Constant']:
        df_agent = df_tc_all[df_tc_all['Agent_Type'] == agent]
        if df_agent.empty: continue
        
        final_pha = df_agent['Total_PHA'].max()
        # 初期ゴム100.0g/Lからの分解量を計算
        initial_rubber = 100.0
        final_rubber = df_agent['Rubber'].iloc[-1]
        degraded_rubber = max(0, initial_rubber - final_rubber)
        
        summary.append({
            'Agent': agent, 
            'Total PHA (mmol)': final_pha, 
            'Degraded Rubber (g/L)': degraded_rubber
        })

    df_summary = pd.DataFrame(summary)
    
    # 2軸プロットでPHAとゴム分解量を比較
    x = np.arange(len(df_summary['Agent']))
    width = 0.35
    
    rects1 = ax_a.bar(x - width/2, df_summary['Total PHA (mmol)'], width, label='Final PHA (mmol)', color='#1f77b4', alpha=0.8)
    
    ax_a2 = ax_a.twinx()
    rects2 = ax_a2.bar(x + width/2, df_summary['Degraded Rubber (g/L)'], width, label='Degraded Rubber (g/L)', color='#d62728', alpha=0.8)
    
    ax_a.set_title('A. Trade-off Analysis: PHA Production vs. Rubber Degradation', loc='left', fontweight='bold')
    ax_a.set_ylabel('PHA Accumulated (mmol)', color='#1f77b4', fontweight='bold')
    ax_a2.set_ylabel('Rubber Degraded (g/L)', color='#d62728', fontweight='bold')
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(df_summary['Agent'])
    
    # 凡例の統合
    lines, labels = ax_a.get_legend_handles_labels()
    lines2, labels2 = ax_a2.get_legend_handles_labels()
    ax_a.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    # 棒の上に数値を表示
    def autolabel(rects, ax, color):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', color=color)
    autolabel(rects1, ax_a, '#1f77b4')
    autolabel(rects2, ax_a2, '#d62728')

except Exception as e:
    ax_a.text(0.5, 0.5, f"Data Error: {e}", ha='center', va='center')

# --- 3B-D 用の共通プロット関数 ---
def plot_timecourse_refined(ax, df_subset, title, max_biomass=10.0, max_pha=120.0):
    if df_subset.empty:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        return []

    # Biomass (Left Axis)
    l1 = ax.plot(df_subset['Time'], df_subset['Biomass_OR16'], label='OR16 (Lcp)', color='#2ca02c', linewidth=2.5)
    l2 = ax.plot(df_subset['Time'], df_subset['Biomass_NS21'], label='NS21 (Rox/PHA)', color='#d62728', linewidth=2.5)
    l3 = ax.plot(df_subset['Time'], df_subset['Biomass_LP'], label='LP (Buffer)', color='#9467bd', linewidth=2)
    ax.set_xlabel('Time (Hours)')
    ax.set_ylabel('Biomass (g/L)', fontweight='bold')
    ax.set_ylim(0, max_biomass)
    
    # pH, PHA & Rubber (Right Axis)
    ax2 = ax.twinx()
    # Optimal pH range (6.5 - 7.5) shaded
    ax2.axhspan(6.5, 7.5, color='green', alpha=0.07, label='Optimal pH Range')
    
    l4 = ax2.plot(df_subset['Time'], df_subset['pH'], label='pH Level', color='#8c564b', linestyle='--', linewidth=1.5, alpha=0.8)
    l5 = ax2.plot(df_subset['Time'], df_subset['Total_PHA'], label='Total PHA', color='gold', linewidth=3, alpha=0.9)
    # ゴムの残量をプロット (初期100を10に圧縮して表示するか、そのまま表示)
    l6 = ax2.plot(df_subset['Time'], df_subset['Rubber'] / 10.0, label='Rubber (x0.1 g/L)', color='black', linestyle=':', linewidth=2, alpha=0.6)
    
    ax2.axhline(y=7.0, color='gray', linestyle=':', linewidth=1.0)
    ax2.set_ylabel('pH / PHA / Rubber', color='black', fontweight='bold')
    ax2.set_ylim(0, max_pha)
    
    ax.set_title(title, loc='left', fontweight='bold', fontsize=14)
    
    return l1 + l2 + l3 + l4 + l5 + l6

# --- 3B-D の描画 ---
if 'df_tc_all' in locals():
    max_bio = df_tc_all[['Biomass_OR16', 'Biomass_NS21', 'Biomass_LP']].max().max() * 1.2
    max_pha = max(df_tc_all['Total_PHA'].max() * 1.1, 15.0)
    right_max = max(max_pha, 15.0) # Rubber/10 (100->10) が収まるように

    ax_b = fig.add_subplot(gs[1, 0])
    lines = plot_timecourse_refined(ax_b, df_tc_all[df_tc_all['Agent_Type'] == 'RL'], 'B. RL Agent (Degradation Focus)', max_bio, right_max)

    ax_c = fig.add_subplot(gs[1, 1])
    plot_timecourse_refined(ax_c, df_tc_all[df_tc_all['Agent_Type'] == 'Dual-PI'], 'C. Dual-PI (Production Focus)', max_bio, right_max)

    ax_d = fig.add_subplot(gs[1, 2])
    plot_timecourse_refined(ax_d, df_tc_all[df_tc_all['Agent_Type'] == 'Constant'], 'D. Constant Feed (Early Crash)', max_bio, right_max)

    # 凡例
    if lines:
        labels = [l.get_label() for l in lines]
        opt_patch = Patch(color='green', alpha=0.1, label='Optimal pH Range')
        ax_b.legend(handles=lines + [opt_patch], loc='upper left', frameon=True, framealpha=0.9, fontsize=8)

plt.tight_layout()
plt.savefig('paper_figures/Figure_3_RL_vs_Baselines.png', bbox_inches='tight')
plt.savefig('paper_figures/Figure_3_RL_vs_Baselines.pdf', bbox_inches='tight')
print("✅ Figure 3 updated with Rubber Degradation analysis.")
