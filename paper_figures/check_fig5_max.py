import pandas as pd
df = pd.read_csv('paper_figures/fig5_product_optimization.csv')
print(f"Max Rubber Remaining: {df['Rubber_Remaining'].max():.2f}")
print(f"Min Rubber Remaining: {df['Rubber_Remaining'].min():.2f}")
print(f"Max C30: {df['C30_oligo_e'].max():.2f}")
print(f"Max ODTD: {df['odtd_e'].max():.2f}")
print(f"Max PHA: {df['Total_PHA'].max():.2f}")
print(f"Survival Time: {df['Time'].max():.2f}")
