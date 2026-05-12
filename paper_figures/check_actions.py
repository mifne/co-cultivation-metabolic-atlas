import pandas as pd
df = pd.read_csv('paper_figures/fig2CD_heatmap.csv')
for i in range(5):
    col = f'Action_{i}'
    print(f"{col}: Min={df[col].min()}, Max={df[col].max()}, Unique={df[col].nunique()}")
