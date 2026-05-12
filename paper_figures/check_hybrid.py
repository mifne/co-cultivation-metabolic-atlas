import pandas as pd
df = pd.read_csv('paper_figures/fig5_hybrid.csv')
print(f"Max PHA: {df['Total_PHA'].max()}")
