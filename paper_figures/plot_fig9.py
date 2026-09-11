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

def plot_figure_9():
    print("Plotting Figure 9 (Metabolic Flux Handoff)...")
    try:
        df = pd.read_csv('paper_figures/fig9_raw_flux_log.csv')
    except Exception as e:
        print(f"Error loading fig9_raw_flux_log.csv: {e}")
        return

    # Filter out empty or initialization rows
    df = df[df['time'] > 0]

    # Extract species specific data
    or16 = df[df['species'].str.contains('OR16', case=False, na=False)].copy()
    ns21 = df[df['species'].str.contains('NS21', case=False, na=False)].copy()

    if or16.empty or ns21.empty:
        print("Warning: Species data not found in flux log.")
        return

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1, hspace=0.3)

    # 9A: Growth Rates Comparison
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(or16['time'], or16['growth_rate'], color='#2ca02c', linewidth=2.5, label='OR16 Specific Growth Rate (1/h)')
    ax_a.plot(ns21['time'], ns21['growth_rate'], color='#d62728', linewidth=2.5, label='NS21 Specific Growth Rate (1/h)')
    
    ax_a.set_ylabel('Growth Rate $\mu$ (1/h)', fontweight='bold')
    ax_a.set_title('A. Co-culture Growth Dynamics', loc='left', fontweight='bold')
    ax_a.legend(loc='upper right')

    # 9B: Intracellular Flux Handoff
    ax_b = fig.add_subplot(gs[1, 0])
    
    # Negative flux means uptake, positive means secretion/production in FBA EX_ reactions usually.
    # Let's plot the absolute values or just the raw logged values.
    # Assuming 'flux_EX_odtd_e' is negative when taken up by NS21. We take absolute value.
    if 'flux_EX_odtd_e' in ns21.columns:
        ns21_odtd_uptake = ns21['flux_EX_odtd_e'].abs()
        ax_b.plot(ns21['time'], ns21_odtd_uptake, color='#1f77b4', linewidth=3, label='NS21 ODTD Uptake Flux (mmol/gDW/h)')
    
    if 'flux_EX_pha_c' in ns21.columns:
        ns21_pha_prod = ns21['flux_EX_pha_c'].abs()
        ax_b.plot(ns21['time'], ns21_pha_prod, color='#9467bd', linewidth=3, linestyle='--', label='NS21 PHA Synthesis Flux (mmol/gDW/h)')

    ax_b.set_xlabel('Time (Hours)', fontweight='bold')
    ax_b.set_ylabel('Metabolic Flux (mmol/gDW/h)', fontweight='bold')
    ax_b.set_title('B. Intracellular Metabolic Flux Handoff (Ground Truth)', loc='left', fontweight='bold')
    
    ax_b.annotate('Flux confirms direct conversion\nfrom intermediate to product',
                  xy=(ns21['time'].max()*0.5, ns21['flux_EX_pha_c'].abs().max() * 0.8),
                  xytext=(ns21['time'].max()*0.6, ns21['flux_EX_pha_c'].abs().max() * 0.9),
                  fontsize=12, style='italic', color='black')

    ax_b.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('paper_figures/Figure_9_Metabolic_Flux.pdf', bbox_inches='tight')
    plt.savefig('paper_figures/Figure_9_Metabolic_Flux.png', bbox_inches='tight')
    print("Saved Figure_9_Metabolic_Flux")

if __name__ == "__main__":
    plot_figure_9()
