import pandas as pd

df = pd.read_csv('data/submodel_outputs/etf_flow.csv', index_col=0)
print('Shape:', df.shape)
print('Date range:', df.index.min(), 'to', df.index.max())

print('\nStats:')
print(df.describe())

print('\nNon-zero counts:')
print('capital_intensity:', (df['etf_capital_intensity'] != 0).sum())
print('pv_divergence:', (df['etf_pv_divergence'] != 0).sum())

print('\nFirst non-zero capital_intensity rows:')
mask = df['etf_capital_intensity'] != 0
if mask.any():
    print(df[mask].head())
else:
    print('All zeros!')

print('\nFirst non-zero pv_divergence rows:')
mask = df['etf_pv_divergence'] != 0
if mask.any():
    print(df[mask].head())
else:
    print('All zeros!')
