import pandas as pd

d1 = {'gene_id': ['G1', 'G2', 'G3'], 'col2': [3, 4, 6], 'col3': [0.3, 0.44, 0.5]}
d2 = {'gene_id': ['G1', 'G2'], 'meth1': [300, 440]}
df1 = pd.DataFrame(data=d1)
df2 = pd.DataFrame(data=d2)

merged_df = df1.merge(df2, on='gene_id', how='inner')

print(merged_df)