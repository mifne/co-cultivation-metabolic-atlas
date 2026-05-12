import pandas as pd
import numpy as np

# Fig 2B
df2 = pd.read_csv('paper_figures/fig2B_timecourse.csv')
ph_mean = df2['pH'].mean()
ph_std = df2['pH'].std()
or16_max = df2['Biomass_OR16'].max()
ns21_max = df2['Biomass_NS21'].max()
lp_max = df2['Biomass_LP'].max()
print(f"--- Fig 2 Stats ---")
print(f"pH Mean: {ph_mean:.3f} ± {ph_std:.3f}")
print(f"Max Biomass -> OR16: {or16_max:.3f}, NS21: {ns21_max:.3f}, LP: {lp_max:.3f}")

# Fig 3A
df3 = pd.read_csv('paper_figures/fig3A_mse_sampling.csv')
mse_12m_rl = df3.loc[df3['Interval_Minutes']==12, 'RL_MSE'].values[0]
mse_12m_pi = df3.loc[df3['Interval_Minutes']==12, 'PI_MSE'].values[0]
mse_60m_rl = df3.loc[df3['Interval_Minutes']==60, 'RL_MSE'].values[0]
mse_60m_pi = df3.loc[df3['Interval_Minutes']==60, 'PI_MSE'].values[0]
print(f"\n--- Fig 3 Stats ---")
print(f"MSE at 12min -> RL: {mse_12m_rl:.4f}, PI: {mse_12m_pi:.4f}")
print(f"MSE at 60min -> RL: {mse_60m_rl:.4f}, PI: {mse_60m_pi:.4f}")

# Fig 4
df4 = pd.read_csv('paper_figures/fig4_survival_evolution.csv')
max_rolling = df4['survival_hours_rolling'].max()
first_10_mean = df4['survival_hours'][:10].mean()
print(f"\n--- Fig 4 Stats ---")
print(f"First 10 episodes mean survival: {first_10_mean:.2f}h")
print(f"Max rolling survival: {max_rolling:.2f}h")

# Fig 5 Hybrid
df5h = pd.read_csv('paper_figures/fig5_hybrid.csv')
c30_peak = df5h['C30_oligo_e'].max()
odtd_peak = df5h['odtd_e'].max()
pha_max = df5h['Total_PHA'].max()
rub_min = df5h['Rubber_Remaining'].min()
print(f"\n--- Fig 5 Hybrid Stats ---")
print(f"Rubber Min: {rub_min:.2f} (Degraded: {100-rub_min:.2f}g)")
print(f"C30 Peak: {c30_peak:.2f} mM")
print(f"ODTD Peak: {odtd_peak:.2f} mM")
print(f"PHA Max: {pha_max:.2f} mmol")

