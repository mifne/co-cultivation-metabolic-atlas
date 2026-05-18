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

def plot_figure_6():
    print("Plotting Figure 6 (Resource Efficiency)...")
    try:
        df_tc = pd.read_csv('paper_figures/fig3B_E_timecourse.csv')
    except Exception as e:
        print(f"Error loading fig3B_E_timecourse.csv: {e}")
        return

    # We use the 12-min interval data for resource efficiency comparison
    df_rl = df_tc[(df_tc['Agent_Type'] == 'RL') & (df_tc['Interval'] == 12)]
    # extract_fig3_data.py uses 'Dual-PI' label (not 'PI')
    df_pi = df_tc[(df_tc['Agent_Type'] == 'Dual-PI') & (df_tc['Interval'] == 12)]
    if df_pi.empty:
        df_pi = df_tc[(df_tc['Agent_Type'] == 'PI') & (df_tc['Interval'] == 12)]

    if df_rl.empty or df_pi.empty:
        print("Warning: Data for Interval 12 not found. Using available data.")
        df_rl = df_tc[df_tc['Agent_Type'] == 'RL'].head(200) if not df_tc[df_tc['Agent_Type'] == 'RL'].empty else df_tc.head(200)
        df_pi = df_tc[df_tc['Agent_Type'].isin(['Dual-PI', 'PI'])].head(200) if not df_tc[df_tc['Agent_Type'].isin(['Dual-PI', 'PI'])].empty else df_tc.head(200)
    if df_rl.empty:
        print("Error: No RL data at all. Aborting Fig 6.")
        return

    # Assuming Action_0, 1, 2, 3 are feeds (specific and common).
    # PI feeds constant 0.05 for 0,1,3. RL dynamically feeds.
    # Total feed used = sum of actions over time
    total_feed_rl = df_rl[['Action_0', 'Action_1', 'Action_2', 'Action_3']].sum().sum()
    total_feed_pi = df_pi[['Action_0', 'Action_1', 'Action_2', 'Action_3']].sum().sum()

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, width_ratios=[1, 2], wspace=0.3)

    # Panel A: Total Feed Comparison
    ax_a = fig.add_subplot(gs[0, 0])
    labels = ['RL Agent', 'PI Controller']
    values = [total_feed_rl, total_feed_pi]
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = ax_a.bar(labels, values, color=colors, alpha=0.8)
    ax_a.set_ylabel('Total Nutrients Fed (Cumulative Action Sum)', fontweight='bold')
    ax_a.set_title('A. Resource Efficiency', loc='left', fontweight='bold')
    
    for bar in bars:
        yval = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.02), f'{yval:.1f}', ha='center', va='bottom', fontweight='bold')

    # Panel B: Feed Action Over Time
    ax_b = fig.add_subplot(gs[0, 1])
    # Sum of all feed actions at each timestep
    feed_rl_time = df_rl['Action_0'] + df_rl['Action_1'] + df_rl['Action_2'] + df_rl['Action_3']
    feed_pi_time = df_pi['Action_0'] + df_pi['Action_1'] + df_pi['Action_2'] + df_pi['Action_3']

    ax_b.plot(df_rl['Time'], feed_rl_time, color='#1f77b4', label='RL Dynamic Feeding', linewidth=2, alpha=0.8)
    ax_b.plot(df_pi['Time'], feed_pi_time, color='#ff7f0e', label='PI Continuous Feeding', linewidth=2, linestyle='--', alpha=0.8)
    
    ax_b.set_xlabel('Time (Hours)', fontweight='bold')
    ax_b.set_ylabel('Feed Rate (Action Sum)', fontweight='bold')
    ax_b.set_title('B. Dynamic Feed Strategy', loc='left', fontweight='bold')
    ax_b.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('paper_figures/Figure_6_Resource_Efficiency.pdf', bbox_inches='tight')
    plt.savefig('paper_figures/Figure_6_Resource_Efficiency.png', bbox_inches='tight')
    print("Saved Figure_6_Resource_Efficiency")

def plot_figure_7():
    print("Plotting Figure 7 (Perturbation Recovery)...")
    try:
        df = pd.read_csv('paper_figures/fig7_shock_test.csv')
    except Exception as e:
        print(f"Error loading fig7_shock_test.csv: {e}")
        return

    df_rl = df[df['Agent_Type'] == 'RL']
    df_pi = df[df['Agent_Type'] == 'PI']

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)

    def plot_shock_panel(ax_ph, ax_bio, df_sub, title_ph, title_bio):
        # pH Plot
        ax_ph.plot(df_sub['Time'], df_sub['pH'], color='#8c564b', linewidth=3)
        ax_ph.axhline(y=7.0, color='gray', linestyle=':', linewidth=2)
        ax_ph.axvline(x=72.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax_ph.annotate('Acid Shock (pH 4.5)', xy=(72, 4.5), xytext=(80, 5.0),
                       arrowprops=dict(facecolor='red', shrink=0.05), color='red', fontweight='bold')
        ax_ph.set_ylim(3, 9)
        ax_ph.set_ylabel('pH Level', fontweight='bold')
        ax_ph.set_title(title_ph, loc='left', fontweight='bold')

        # Biomass Plot
        ax_bio.plot(df_sub['Time'], df_sub['Biomass_OR16'], label='OR16 (Lcp)', color='#2ca02c', linewidth=2.5)
        ax_bio.plot(df_sub['Time'], df_sub['Biomass_NS21'], label='NS21 (Rox)', color='#d62728', linewidth=2.5)
        ax_bio.plot(df_sub['Time'], df_sub['Biomass_LP'], label='LP (Buffer)', color='#9467bd', linewidth=2.5)
        ax_bio.axvline(x=72.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax_bio.set_ylabel('Biomass (g/L)', fontweight='bold')
        ax_bio.set_xlabel('Time (Hours)', fontweight='bold')
        ax_bio.set_title(title_bio, loc='left', fontweight='bold')
        ax_bio.legend(loc='upper right')

    # Top Row: RL
    ax_rl_ph = fig.add_subplot(gs[0, 0])
    ax_rl_bio = fig.add_subplot(gs[0, 1])
    plot_shock_panel(ax_rl_ph, ax_rl_bio, df_rl, 'A. RL Agent: pH Recovery', 'B. RL Agent: Biomass Survival')

    # Bottom Row: PI
    ax_pi_ph = fig.add_subplot(gs[1, 0])
    ax_pi_bio = fig.add_subplot(gs[1, 1])
    plot_shock_panel(ax_pi_ph, ax_pi_bio, df_pi, 'C. PI Control: pH Collapse', 'D. PI Control: System Failure')

    plt.tight_layout()
    plt.savefig('paper_figures/Figure_7_Shock_Recovery.pdf', bbox_inches='tight')
    plt.savefig('paper_figures/Figure_7_Shock_Recovery.png', bbox_inches='tight')
    print("Saved Figure_7_Shock_Recovery")

if __name__ == "__main__":
    plot_figure_6()
    plot_figure_7()
