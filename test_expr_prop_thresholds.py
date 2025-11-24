#!/usr/bin/env python3
"""
Test different expr_prop thresholds for CCC analysis
Focus on Glut->GABA and GABA->Glut in HIP->ACC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Paths
liana_dir = Path("/data/CARD_singlecell/Ellie_finalized/analyses/2024_analyses/240930_LIANA")
results_dir = liana_dir / "expr_prop_test"
results_dir.mkdir(exist_ok=True)

# Load the full results
print("Loading LIANA results...")
df = pd.read_csv(liana_dir / "liana_results_complete.csv")

# Filter for HIP->ACC
print("Filtering for HIP->ACC...")
df_filtered = df[
    (df['source_region'] == 'HIP') & 
    (df['target_region'] == 'ACC')
].copy()

print(f"Total interactions in HIP->ACC: {len(df_filtered)}")

# Test different expr_prop thresholds
expr_prop_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Cell type pairs to test
cell_pairs = [
    ('Glut', 'GABA'),
    ('GABA', 'Glut')
]

# Create figure for p-value distributions
fig, axes = plt.subplots(len(cell_pairs), len(expr_prop_thresholds), 
                         figsize=(20, 8), sharex=True, sharey=True)

results_summary = []

for row_idx, (source_ct, target_ct) in enumerate(cell_pairs):
    # Filter for this cell type pair
    pair_data = df_filtered[
        (df_filtered['source'] == source_ct) & 
        (df_filtered['target'] == target_ct)
    ].copy()
    
    print(f"\n{source_ct} -> {target_ct}:")
    print(f"Total interactions before filtering: {len(pair_data)}")
    
    for col_idx, threshold in enumerate(expr_prop_thresholds):
        ax = axes[row_idx, col_idx]
        
        # Apply expr_prop filter
        filtered_data = pair_data[
            (pair_data['expr_prop'] >= threshold)
        ].copy()
        
        n_interactions = len(filtered_data)
        
        if n_interactions > 0:
            pvals = filtered_data['pvalue'].values
            
            # Remove NaN p-values
            pvals_clean = pvals[~np.isnan(pvals)]
            
            # Plot histogram
            ax.hist(pvals_clean, bins=30, edgecolor='black', alpha=0.7)
            
            # Add uniform distribution reference line
            ax.axhline(y=len(pvals_clean)/30, color='red', linestyle='--', 
                      linewidth=1, alpha=0.5, label='Uniform')
            
            # Calculate some statistics
            if len(pvals_clean) > 0:
                n_sig = np.sum(pvals_clean < 0.05)
                prop_sig = n_sig / len(pvals_clean)
                
                # KS test for uniformity
                ks_stat, ks_pval = stats.kstest(pvals_clean, 'uniform')
                
                results_summary.append({
                    'source': source_ct,
                    'target': target_ct,
                    'expr_prop_threshold': threshold,
                    'n_interactions': n_interactions,
                    'n_with_pval': len(pvals_clean),
                    'n_sig_0.05': n_sig,
                    'prop_sig': prop_sig,
                    'ks_stat': ks_stat,
                    'ks_pval': ks_pval,
                    'median_pval': np.median(pvals_clean),
                    'mean_pval': np.mean(pvals_clean)
                })
            else:
                results_summary.append({
                    'source': source_ct,
                    'target': target_ct,
                    'expr_prop_threshold': threshold,
                    'n_interactions': n_interactions,
                    'n_with_pval': 0,
                    'n_sig_0.05': 0,
                    'prop_sig': 0,
                    'ks_stat': np.nan,
                    'ks_pval': np.nan,
                    'median_pval': np.nan,
                    'mean_pval': np.nan
                })
        else:
            results_summary.append({
                'source': source_ct,
                'target': target_ct,
                'expr_prop_threshold': threshold,
                'n_interactions': 0,
                'n_with_pval': 0,
                'n_sig_0.05': 0,
                'prop_sig': 0,
                'ks_stat': np.nan,
                'ks_pval': np.nan,
                'median_pval': np.nan,
                'mean_pval': np.nan
            })
        
        # Formatting
        if row_idx == 0:
            ax.set_title(f'expr_prop ≥ {threshold}', fontsize=10, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel(f'{source_ct}→{target_ct}\nFrequency', fontsize=9)
        if row_idx == len(cell_pairs) - 1:
            ax.set_xlabel('P-value', fontsize=9)
        
        # Add text with n
        ax.text(0.98, 0.98, f'n={n_interactions}', 
               transform=ax.transAxes, 
               ha='right', va='top',
               fontsize=8,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(results_dir / 'pvalue_distributions_by_expr_prop.png', 
           dpi=300, bbox_inches='tight')
print(f"\nSaved p-value distribution plot")

# Create summary dataframe
summary_df = pd.DataFrame(results_summary)
summary_df.to_csv(results_dir / 'expr_prop_threshold_summary.csv', index=False)
print(f"\nSaved summary statistics")

# Create a summary plot showing how metrics change with threshold
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, (source_ct, target_ct) in enumerate(cell_pairs):
    pair_summary = summary_df[
        (summary_df['source'] == source_ct) & 
        (summary_df['target'] == target_ct)
    ]
    
    # Plot 1: Number of interactions
    axes[0, 0].plot(pair_summary['expr_prop_threshold'], 
                    pair_summary['n_interactions'], 
                    marker='o', label=f'{source_ct}→{target_ct}')
    
    # Plot 2: Proportion significant
    axes[0, 1].plot(pair_summary['expr_prop_threshold'], 
                    pair_summary['prop_sig'], 
                    marker='o', label=f'{source_ct}→{target_ct}')
    
    # Plot 3: KS test p-value (for uniformity test)
    axes[1, 0].plot(pair_summary['expr_prop_threshold'], 
                    pair_summary['ks_pval'], 
                    marker='o', label=f'{source_ct}→{target_ct}')
    
    # Plot 4: Median p-value
    axes[1, 1].plot(pair_summary['expr_prop_threshold'], 
                    pair_summary['median_pval'], 
                    marker='o', label=f'{source_ct}→{target_ct}')

axes[0, 0].set_xlabel('expr_prop threshold')
axes[0, 0].set_ylabel('Number of interactions')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].set_xlabel('expr_prop threshold')
axes[0, 1].set_ylabel('Proportion significant (p<0.05)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5)

axes[1, 0].set_xlabel('expr_prop threshold')
axes[1, 0].set_ylabel('KS test p-value\n(uniformity test)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
axes[1, 0].set_yscale('log')

axes[1, 1].set_xlabel('expr_prop threshold')
axes[1, 1].set_ylabel('Median p-value')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(results_dir / 'threshold_summary_metrics.png', 
           dpi=300, bbox_inches='tight')
print(f"\nSaved threshold summary plot")

# Print summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(summary_df.to_string(index=False))
