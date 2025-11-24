#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
output_dir = Path("/scratch/easmit31/cell_cell/results/expr_prop_test/analysis")
lm_df = pd.read_csv(output_dir / 'lm_results_all_thresholds.csv')

thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
cell_pairs = [
    ('Glutamatergic', 'GABA'),
    ('GABA', 'Glutamatergic')
]

print("Creating p-value distribution plots with labels on all subplots...")

# 1. AGE P-VALUE DISTRIBUTIONS
fig, axes = plt.subplots(len(cell_pairs), len(thresholds), 
                         figsize=(24, 8), sharex=True)

for row_idx, (source_ct, target_ct) in enumerate(cell_pairs):
    for col_idx, threshold in enumerate(thresholds):
        ax = axes[row_idx, col_idx]
        
        pair_lm = lm_df[
            (lm_df['expr_prop_threshold'] == threshold) &
            (lm_df['source'] == source_ct) &
            (lm_df['target'] == target_ct)
        ]
        
        if len(pair_lm) > 0:
            pvals = pair_lm['age_pval'].values
            pvals_clean = pvals[~np.isnan(pvals)]
            
            if len(pvals_clean) > 0:
                # Plot histogram
                ax.hist(pvals_clean, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                
                # Add uniform reference line
                expected_height = len(pvals_clean) / 30
                ax.axhline(y=expected_height, color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.7, label='Uniform')
                
                n_sig = np.sum(pvals_clean < 0.05)
                ax.text(0.98, 0.98, f'n={len(pair_lm)}\nsig={n_sig}', 
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=8, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Title on ALL subplots
        ax.set_title(f'{source_ct}→{target_ct}, expr_prop ≥ {threshold}', 
                    fontsize=10, fontweight='bold')
        
        # X-axis label on ALL subplots
        ax.set_xlabel('Age P-value', fontsize=10)
        
        # Y-axis label
        ax.set_ylabel('Frequency', fontsize=10)

plt.suptitle('Age Effect P-value Distributions', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'age_pvalue_distributions_alllabels.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'age_pvalue_distributions_alllabels.png'}")

# 2. SEX P-VALUE DISTRIBUTIONS
fig, axes = plt.subplots(len(cell_pairs), len(thresholds), 
                         figsize=(24, 8), sharex=True)

for row_idx, (source_ct, target_ct) in enumerate(cell_pairs):
    for col_idx, threshold in enumerate(thresholds):
        ax = axes[row_idx, col_idx]
        
        pair_lm = lm_df[
            (lm_df['expr_prop_threshold'] == threshold) &
            (lm_df['source'] == source_ct) &
            (lm_df['target'] == target_ct) &
            (lm_df['has_sex_variance'])
        ]
        
        if len(pair_lm) > 0:
            pvals = pair_lm['sex_pval'].values
            pvals_clean = pvals[~np.isnan(pvals)]
            
            if len(pvals_clean) > 0:
                # Plot histogram
                ax.hist(pvals_clean, bins=30, edgecolor='black', alpha=0.7, color='coral')
                
                # Add uniform reference line
                expected_height = len(pvals_clean) / 30
                ax.axhline(y=expected_height, color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.7)
                
                n_sig = np.sum(pvals_clean < 0.05)
                ax.text(0.98, 0.98, f'n={len(pair_lm)}\nsig={n_sig}', 
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=8, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # Title on ALL subplots
        ax.set_title(f'{source_ct}→{target_ct}, expr_prop ≥ {threshold}', 
                    fontsize=10, fontweight='bold')
        
        # X-axis label on ALL subplots
        ax.set_xlabel('Sex P-value', fontsize=10)
        
        # Y-axis label
        ax.set_ylabel('Frequency', fontsize=10)

plt.suptitle('Sex Effect P-value Distributions', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'sex_pvalue_distributions_alllabels.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'sex_pvalue_distributions_alllabels.png'}")

print("\nDone!")
