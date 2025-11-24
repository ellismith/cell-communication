#!/usr/bin/env python3
"""
Test for sex differences in L-R pair expression across individuals
"""
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
import os
import gc

results_dir = "results/u01_per_individual"
output_dir = "results/u01_sex_differences"
os.makedirs(output_dir, exist_ok=True)

result_files = [f for f in os.listdir(results_dir) 
                if f.endswith('_results.csv') and 'ALL_INTERACTIONS' not in f]

print(f"Found {len(result_files)} individuals")

# Load only significant interactions
print("Loading only significant interactions (q<0.05)...")
sig_data = []
for idx, result_file in enumerate(result_files):
    if idx % 10 == 0:
        print(f"  Processing {idx}/{len(result_files)}...")
    
    df = pd.read_csv(f"{results_dir}/{result_file}")
    df_sig = df[df['qval'] < 0.05].copy()
    
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

# Parse
combined_sig['combo'] = combined_sig['source'] + ' -> ' + combined_sig['target']
combined_sig['lr_pair'] = combined_sig['ligand_complex'] + ' -> ' + combined_sig['receptor_complex']

# Count individuals per sex
n_male = combined_sig[combined_sig['sex'] == 'M']['animal_id'].nunique()
n_female = combined_sig[combined_sig['sex'] == 'F']['animal_id'].nunique()
print(f"\nIndividuals: {n_male} males, {n_female} females")

print("\nTesting sex differences...")

results_list = []

grouped = combined_sig.groupby(['combo', 'lr_pair'])
total_groups = len(grouped)

for idx, ((combo, lr_pair), group) in enumerate(grouped):
    if idx % 100 == 0:
        print(f"  Testing {idx}/{total_groups}...")
    
    # Need at least 5 individuals per sex
    male_data = group[group['sex'] == 'M']['lr_means'].values
    female_data = group[group['sex'] == 'F']['lr_means'].values
    
    if len(male_data) < 5 or len(female_data) < 5:
        continue
    
    # Remove NaN
    male_clean = male_data[~pd.isna(male_data)]
    female_clean = female_data[~pd.isna(female_data)]
    
    if len(male_clean) < 5 or len(female_clean) < 5:
        continue
    
    # T-test and Mann-Whitney U test
    t_stat, t_pval = ttest_ind(male_clean, female_clean)
    u_stat, u_pval = mannwhitneyu(male_clean, female_clean, alternative='two-sided')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((male_clean.std()**2 + female_clean.std()**2) / 2)
    cohens_d = (male_clean.mean() - female_clean.mean()) / pooled_std if pooled_std > 0 else 0
    
    # Fold change
    if female_clean.mean() > 0:
        fold_change = male_clean.mean() / female_clean.mean()
    else:
        fold_change = np.nan
    
    results_list.append({
        'combo': combo,
        'lr_pair': lr_pair,
        'n_male': len(male_clean),
        'n_female': len(female_clean),
        'male_mean': male_clean.mean(),
        'female_mean': female_clean.mean(),
        'male_std': male_clean.std(),
        'female_std': female_clean.std(),
        'fold_change': fold_change,
        'cohens_d': cohens_d,
        't_stat': t_stat,
        't_pval': t_pval,
        'mann_whitney_u': u_stat,
        'mann_whitney_p': u_pval
    })

results_df = pd.DataFrame(results_list)

print(f"\nTested {len(results_df)} combo-pathway pairs")

# FDR correction
_, qvals_t, _, _ = multipletests(results_df['t_pval'], method='fdr_bh')
results_df['t_qval'] = qvals_t

_, qvals_u, _, _ = multipletests(results_df['mann_whitney_p'], method='fdr_bh')
results_df['mann_whitney_q'] = qvals_u

# Sort by effect size
results_df['abs_cohens_d'] = results_df['cohens_d'].abs()
results_df = results_df.sort_values('abs_cohens_d', ascending=False)

# Save
results_df.to_csv(f"{output_dir}/sex_differences_results_all.csv", index=False)
print(f"\n✓ Saved to {output_dir}/sex_differences_results_all.csv")

sig_results = results_df[results_df['t_qval'] < 0.05].copy()
sig_results.to_csv(f"{output_dir}/sex_differences_results_significant.csv", index=False)
print(f"✓ Saved {len(sig_results)} significant sex differences")

# Summary
print(f"\n{'='*70}")
print(f"Total tested: {len(results_df)}")
print(f"Significant (FDR q<0.05): {len(sig_results)}")

if len(sig_results) > 0:
    print(f"\nTop 10 MALE-biased (higher in males):")
    male_biased = sig_results[sig_results['cohens_d'] > 0].sort_values('cohens_d', ascending=False)
    if len(male_biased) > 0:
        print(male_biased[['combo', 'lr_pair', 'male_mean', 'female_mean', 'cohens_d', 't_qval']].head(10).to_string(index=False))
    
    print(f"\nTop 10 FEMALE-biased (higher in females):")
    female_biased = sig_results[sig_results['cohens_d'] < 0].sort_values('cohens_d')
    if len(female_biased) > 0:
        print(female_biased[['combo', 'lr_pair', 'male_mean', 'female_mean', 'cohens_d', 't_qval']].head(10).to_string(index=False))

print(f"\n{'='*70}")
