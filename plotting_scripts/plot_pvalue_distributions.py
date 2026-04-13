#!/usr/bin/env python3
"""
Create p-value distribution plots for age and sex effects
Shows histograms of p-values for all regression results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_file', required=True, help='Path to regression results CSV')
args = parser.parse_args()

results_file = Path(args.results_file)

if not results_file.exists():
    print(f"ERROR: File not found: {results_file}")
    exit(1)

# Load results
df = pd.read_csv(results_file)
print(f"Loaded {len(df):,} tests from {results_file.name}")

# Extract region from filename
filename = results_file.stem
region = filename.replace("whole_", "").split("_age_sex_regression")[0].upper()

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Age p-value distribution
ax = axes[0]
age_pvals = df['age_pval'].dropna()
ax.hist(age_pvals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')

expected_count = len(age_pvals) / 50
ax.axhline(expected_count, color='red', linestyle='--', linewidth=2,
           label=f'Uniform expectation ({expected_count:.0f})')

n_sig_05 = (df['age_pval'] < 0.05).sum()
n_sig_01 = (df['age_pval'] < 0.01).sum()
n_qval_05 = (df['age_qval'] < 0.05).sum()
n_qval_10 = (df['age_qval'] < 0.10).sum()

textstr = f'p < 0.05: {n_sig_05}\n'
textstr += f'p < 0.01: {n_sig_01}\n'
textstr += f'q < 0.05: {n_qval_05}\n'
textstr += f'q < 0.10: {n_qval_10}'

ax.text(0.98, 0.97, textstr, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontsize=10, family='monospace')

ax.set_xlabel('Age P-value', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Age Effect P-value Distribution', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Sex p-value distribution
ax = axes[1]
sex_pvals = df['sex_pval'].dropna()
ax.hist(sex_pvals, bins=50, edgecolor='black', alpha=0.7, color='coral')

expected_count = len(sex_pvals) / 50
ax.axhline(expected_count, color='red', linestyle='--', linewidth=2,
           label=f'Uniform expectation ({expected_count:.0f})')

n_sig_05 = (df['sex_pval'] < 0.05).sum()
n_sig_01 = (df['sex_pval'] < 0.01).sum()
n_qval_05 = (df['sex_qval'] < 0.05).sum()
n_qval_10 = (df['sex_qval'] < 0.10).sum()

textstr = f'p < 0.05: {n_sig_05}\n'
textstr += f'p < 0.01: {n_sig_01}\n'
textstr += f'q < 0.05: {n_qval_05}\n'
textstr += f'q < 0.10: {n_qval_10}'

ax.text(0.98, 0.97, textstr, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontsize=10, family='monospace')

ax.set_xlabel('Sex P-value', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Sex Effect P-value Distribution', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle(f'{region} (n={len(df):,} tests)',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()

output_file = results_file.parent / f'{filename}_pvalue_dist.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")

plt.close()
