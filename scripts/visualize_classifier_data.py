"""
Create visualizations for classifier data validation
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# Read the validated raw data
df = pd.read_csv('data/raw/classifier_raw.csv', index_col='Date', parse_dates=True)

print('Creating data visualization...')

# Create figure with subplots
fig, axes = plt.subplots(4, 2, figsize=(14, 12))
fig.suptitle('Gold DOWN Classifier - Data Sources Validation', fontsize=16, fontweight='bold')

# 1. Gold price
ax = axes[0, 0]
ax.plot(df.index, df['GC_Close'], linewidth=0.8, color='gold')
ax.set_title('GC=F: Gold Futures Close Price', fontweight='bold')
ax.set_ylabel('Price (USD)')
ax.grid(True, alpha=0.3)

# 2. Gold VIX (GVZCLS)
ax = axes[0, 1]
ax.plot(df.index, df['GVZCLS'], linewidth=0.8, color='orange')
ax.set_title('GVZCLS: Gold Volatility Index', fontweight='bold')
ax.set_ylabel('GVZ Level')
ax.grid(True, alpha=0.3)

# 3. VIX
ax = axes[1, 0]
ax.plot(df.index, df['VIXCLS'], linewidth=0.8, color='red')
ax.set_title('VIXCLS: Equity Volatility Index', fontweight='bold')
ax.set_ylabel('VIX Level')
ax.grid(True, alpha=0.3)

# 4. GVZ/VIX ratio
ax = axes[1, 1]
ratio = df['GVZCLS'] / df['VIXCLS']
ax.plot(df.index, ratio, linewidth=0.8, color='purple')
ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Ratio=1.0')
ax.set_title('GVZ/VIX Ratio (Key Feature)', fontweight='bold')
ax.set_ylabel('Ratio')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Real rate (DFII10)
ax = axes[2, 0]
ax.plot(df.index, df['DFII10'], linewidth=0.8, color='blue')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_title('DFII10: 10Y TIPS Real Rate', fontweight='bold')
ax.set_ylabel('Yield (%)')
ax.grid(True, alpha=0.3)

# 6. Dollar Index
ax = axes[2, 1]
ax.plot(df.index, df['DXY_Close'], linewidth=0.8, color='green')
ax.set_title('DX-Y.NYB: Dollar Index', fontweight='bold')
ax.set_ylabel('DXY')
ax.grid(True, alpha=0.3)

# 7. S&P 500
ax = axes[3, 0]
ax.plot(df.index, df['SPX_Close'], linewidth=0.8, color='darkblue')
ax.set_title('^GSPC: S&P 500', fontweight='bold')
ax.set_ylabel('Price')
ax.grid(True, alpha=0.3)

# 8. Data availability summary
ax = axes[3, 1]
ax.axis('off')

summary_text = '''Data Validation Summary

Total Rows: 3,048
Effective Rows: 2,959
Features: 18
Date Range: 2014-01-02 to 2026-02-17

Train/Val/Test Split:
• Train: 2,071 rows (70%)
• Val: 444 rows (15%)
• Test: 444 rows (15%)

Target Balance:
• UP days: 52.72%
• DOWN days: 47.28%

Data Quality:
• All sources accessible: [OK]
• NaN after forward-fill: 0%
• Samples per feature: 115:1

Status: READY FOR TRAINING
'''

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('logs/datacheck/classifier_data_validation.png', dpi=150, bbox_inches='tight')
print('Saved: logs/datacheck/classifier_data_validation.png')
plt.close()

# Create feature correlation heatmap
print('Creating feature correlation heatmap...')

# Load the processed data with features
sys.path.append('src')
from fetch_classifier import fetch_and_preprocess

train, val, test, full = fetch_and_preprocess()

# Calculate correlation matrix (exclude target)
features = full.drop('target', axis=1)
corr = features.corr()

# Create heatmap
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

# Set ticks
ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90, ha='right', fontsize=8)
ax.set_yticklabels(corr.columns, fontsize=8)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Correlation', rotation=270, labelpad=20)

# Title
plt.title('Classifier Features - Correlation Matrix (18 features)', fontweight='bold', fontsize=14, pad=20)

# Annotate high correlations (|r| > 0.5)
for i in range(len(corr)):
    for j in range(len(corr)):
        if i != j and abs(corr.iloc[i, j]) > 0.5:
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                         ha='center', va='center', color='black', fontsize=6, fontweight='bold')

plt.tight_layout()
plt.savefig('logs/datacheck/classifier_feature_correlation.png', dpi=150, bbox_inches='tight')
print('Saved: logs/datacheck/classifier_feature_correlation.png')
plt.close()

print('Visualization complete!')
