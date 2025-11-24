#!/usr/bin/env python3
"""
Test for age correlations in L-R pair expression across individuals
REGION-SPECIFIC VERSION - each individual contributes only 1 data point per region combo
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
import os
import gc

results_dir = "results/u01_per_individual"
output_dir = "results/u01_age_correlations_by_region"
os.makedirs(output_dir, exist_ok=True)

result_files = [f for f in os.listdir(results_dir) 
                if f.endswith('_results.csv') and 'ALL_INTERACTIONS' not in f]

print(f"Found {len(result_files)} individuals")

# Process only significant interactions to reduce memory
print("Loading only significant interactions (q<0.05)...")
sig_data = []
for idx, result_file in enumerate(result_files):
    if idx % 10 == 0:
        print(f"  Processing {idx}/{len(result_files)}...")
    
    df = pd.read_csv(f"{results_dir}/{result_file}")
    df_sig = df[df['qval'] < 0.05].copy()
    
    # Keep only needed columns
    df_sig = df_sig[['source', 'target', 'ligand_complex', 'receptor_complex', 
                     'lr_means', 'animal_id', 'age', 'sex']]
    sig_data.append(df_sig)
    
    del df, df_sig
    if idx % 20 == 0:
        gc.collect()

combined_sig = pd.concat(sig_data, ignore_index=True)
del sig_data
gc.collect()

print(f"Loaded {len(combined_sig):,} significant interactions")

# Parse cell types and regions
combined_sig['source_celltype'] = combined_sig['source'].str.rsplit('_', n=1).str[0]
combined_sig['target_celltype'] = combined_sig['target'].str.rsplit('_', n=1).str[0]
combined_sig['source_region'] = combined_sig['source'].str.rsplit('_', n=1).str[1]
combined_sig['target_region'] = combined_sig['target'].str.rsplit('_', n=1).str[1]

# Create identifiers
combined_sig['celltype_pair'] = combined_sig['source_celltype'] + ' -> ' + combined_sig['target_celltype']
combined_sig['region_pair'] = combined_sig['source_region'] + ' -> ' + combined_sig['target_region']
combined_sig['lr_pair'] = combined_sig['ligand_complex'] + ' -> ' + combined_sig['receptor_complex']

# Classify as within-region or cross-region
combined_sig['interaction_type'] = combined_sig.apply(
    lambda x: 'within' if x['source_region'] == x['target_region'] else 'cross', axis=1
)

print("\nTesting age correlations (region-specific)...")

results_list = []

# Group by: region_pair, celltype_pair, and L-R pair
# This ensures each individual contributes only 1 data point per group
grouped = combined_sig.groupby(['region_pair', 'celltype_pair', 'lr_pair', 'interaction_type'])
total_groups = len(grouped)

for idx, ((region_pair, celltype_pair, lr_pair, interaction_type), group) in enumerate(grouped):
    if idx % 500 == 0:
        print(f"  Testing {idx}/{total_groups}...")
    
    # Check that each individual appears at most once
    n_unique_individuals = group['animal_id'].nunique()
    n_observations = len(group)
    
    if n_unique_individuals != n_observations:
        print(f"WARNING: {region_pair} | {celltype_pair} | {lr_pair} has duplicate individuals!")
        # Keep only one observation per individual (take mean if duplicates exist)
        group = group.groupby('animal_id').agg({
            'age': 'first',
            'sex': 'first',
            'lr_means': 'mean'
        }).reset_index()
    
    # Need at least 10 individuals for meaningful correlation
    if len(group) < 10:
        continue
    
    # Test correlation with age
    age_values = group['age'].values
    lr_means_values = group['lr_means'].values
    
    # Remove NaN
    mask = ~(pd.isna(age_values) | pd.isna(lr_means_values))
    age_clean = age_values[mask]
    lr_clean = lr_means_values[mask]
    
    if len(age_clean) < 10:
        continue
    
    # Correlations
    pearson_r, pearson_p = pearsonr(age_clean, lr_clean)
    spearman_r, spearman_p = spearmanr(age_clean, lr_clean)
    
    results_list.append({
        'region_pair': region_pair,
        'celltype_pair': celltype_pair,
        'lr_pair': lr_pair,
        'interaction_type': interaction_type,
        'n_individuals': len(age_clean),
        'mean_lr_means': lr_clean.mean(),
        'std_lr_means': lr_clean.std(),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p
    })

results_df = pd.DataFrame(results_list)

print(f"\nTested {len(results_df)} region-specific combo-pathway pairs")

# FDR correction separately for within-region and cross-region
print("\nApplying FDR correction separately for within vs cross-region...")

within_df = results_df[results_df['interaction_type'] == 'within'].copy()
cross_df = results_df[results_df['interaction_type'] == 'cross'].copy()

# Within-region FDR
if len(within_df) > 0:
    _, qvals_p_within, _, _ = multipletests(within_df['pearson_p'], method='fdr_bh')
    within_df['pearson_q'] = qvals_p_within
    _, qvals_s_within, _, _ = multipletests(within_df['spearman_p'], method='fdr_bh')
    within_df['spearman_q'] = qvals_s_within

# Cross-region FDR
if len(cross_df) > 0:
    _, qvals_p_cross, _, _ = multipletests(cross_df['pearson_p'], method='fdr_bh')
    cross_df['pearson_q'] = qvals_p_cross
    _, qvals_s_cross, _, _ = multipletests(cross_df['spearman_p'], method='fdr_bh')
    cross_df['spearman_q'] = qvals_s_cross

# Combine back
results_df = pd.concat([within_df, cross_df], ignore_index=True)

# Sort by abs correlation
results_df['abs_pearson_r'] = results_df['pearson_r'].abs()
results_df = results_df.sort_values('abs_pearson_r', ascending=False)

# Save all results
results_df.to_csv(f"{output_dir}/age_correlation_results_all.csv", index=False)
print(f"\n✓ Saved to {output_dir}/age_correlation_results_all.csv")

# Save significant results
sig_results = results_df[results_df['pearson_q'] < 0.05].copy()
sig_results.to_csv(f"{output_dir}/age_correlation_results_significant.csv", index=False)
print(f"✓ Saved {len(sig_results)} significant correlations (FDR q<0.05)")

# Save separately by interaction type
sig_within = sig_results[sig_results['interaction_type'] == 'within'].copy()
sig_cross = sig_results[sig_results['interaction_type'] == 'cross'].copy()

sig_within.to_csv(f"{output_dir}/age_correlation_significant_WITHIN_region.csv", index=False)
sig_cross.to_csv(f"{output_dir}/age_correlation_significant_CROSS_region.csv", index=False)

print(f"  - {len(sig_within)} within-region")
print(f"  - {len(sig_cross)} cross-region")

# Summary
print(f"\n{'='*70}")
print(f"WITHIN-REGION INTERACTIONS:")
print(f"  Total tested: {len(within_df)}")
print(f"  Significant (FDR q<0.05): {len(sig_within)}")

if len(sig_within) > 0:
    print(f"\n  Top 5 POSITIVE (increase with age):")
    pos = sig_within[sig_within['pearson_r'] > 0].sort_values('pearson_r', ascending=False)
    if len(pos) > 0:
        print(pos[['region_pair', 'celltype_pair', 'lr_pair', 'pearson_r', 'pearson_q']].head(5).to_string(index=False))
    
    print(f"\n  Top 5 NEGATIVE (decrease with age):")
    neg = sig_within[sig_within['pearson_r'] < 0].sort_values('pearson_r')
    if len(neg) > 0:
        print(neg[['region_pair', 'celltype_pair', 'lr_pair', 'pearson_r', 'pearson_q']].head(5).to_string(index=False))

print(f"\n{'='*70}")
print(f"CROSS-REGION INTERACTIONS:")
print(f"  Total tested: {len(cross_df)}")
print(f"  Significant (FDR q<0.05): {len(sig_cross)}")

if len(sig_cross) > 0:
    print(f"\n  Top 5 POSITIVE (increase with age):")
    pos = sig_cross[sig_cross['pearson_r'] > 0].sort_values('pearson_r', ascending=False)
    if len(pos) > 0:
        print(pos[['region_pair', 'celltype_pair', 'lr_pair', 'pearson_r', 'pearson_q']].head(5).to_string(index=False))
    
    print(f"\n  Top 5 NEGATIVE (decrease with age):")
    neg = sig_cross[sig_cross['pearson_r'] < 0].sort_values('pearson_r')
    if len(neg) > 0:
        print(neg[['region_pair', 'celltype_pair', 'lr_pair', 'pearson_r', 'pearson_q']].head(5).to_string(index=False))

print(f"\n{'='*70}")
print("✓ Analysis complete!")
