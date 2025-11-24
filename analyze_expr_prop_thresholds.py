#!/usr/bin/env python3
"""
Analyze results across different expr_prop thresholds
Run LM models: lr_means ~ age + sex
Keep full source-target region pairs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Directories
base_results_dir = Path("/scratch/easmit31/cell_cell/results/expr_prop_test")
output_dir = Path("/scratch/easmit31/cell_cell/results/expr_prop_test/analysis")
output_dir.mkdir(exist_ok=True)

# Thresholds to analyze
thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# All four HIP-ACC pairs
target_pairs = [
    ('Glutamatergic_HIP', 'GABA_ACC'),
    ('GABA_HIP', 'Glutamatergic_ACC'),
    ('Glutamatergic_HIP', 'Glutamatergic_ACC'),
    ('GABA_HIP', 'GABA_ACC')
]

print("="*80)
print("LOADING DATA")
print("="*80)

# Load results for each threshold
all_results = {}
for threshold in thresholds:
    suffix = f"_filtering{threshold}".replace('.', 'p')
    results_dir = base_results_dir / f"results{suffix}"
    
    dfs = []
    for csv_file in results_dir.glob("*_results*.csv"):
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    if len(dfs) > 0:
        combined = pd.concat(dfs, ignore_index=True)
        all_results[threshold] = combined
        print(f"Threshold {threshold}: {len(combined):,} interactions, {combined['animal_id'].nunique()} individuals")

print("\n" + "="*80)
print("RUNNING LINEAR MODELS: lr_means ~ age + sex")
print("="*80)

all_lm_results = []

for threshold in thresholds:
    print(f"\nThreshold {threshold}:")
    df = all_results[threshold]
    
    for source, target in target_pairs:
        pair_data = df[(df['source'] == source) & (df['target'] == target)].copy()
        
        if len(pair_data) == 0:
            continue
        
        print(f"  {source}→{target}: {len(pair_data)} rows")
        
        # Create LR pair identifier
        pair_data['LR_pair'] = pair_data['ligand_complex'] + '_' + pair_data['receptor_complex']
        
        # Pivot to wide format - USING lr_means
        df_wide = pair_data.pivot_table(
            index='animal_id',
            columns='LR_pair',
            values='lr_means',
            aggfunc='first'
        )
        
        # Get metadata
        metadata = pair_data[['animal_id', 'age', 'sex']].drop_duplicates().set_index('animal_id')
        df_model = df_wide.merge(metadata, left_index=True, right_index=True, how='left')
        
        # Encode sex
        df_model['sex_binary'] = (df_model['sex'] == 'M').astype(int)
        
        # Get LR columns
        lr_columns = [col for col in df_model.columns if col not in ['age', 'sex', 'sex_binary']]
        
        print(f"    Running {len(lr_columns)} linear models...")
        
        # Run models
        for lr_pair in lr_columns:
            data = df_model[['age', 'sex_binary', lr_pair]].dropna()
            
            if len(data) < 3:
                continue
            
            try:
                has_sex_variance = data['sex_binary'].nunique() == 2
                
                if has_sex_variance:
                    # Model: lr_means ~ age + sex
                    X = data[['age', 'sex_binary']].values
                    y = data[lr_pair].values
                    X_with_const = sm.add_constant(X)
                    model = sm.OLS(y, X_with_const).fit()
                    
                    all_lm_results.append({
                        'expr_prop_threshold': threshold,
                        'source': source,
                        'target': target,
                        'LR_pair': lr_pair,
                        'n_samples': len(data),
                        'has_sex_variance': True,
                        'intercept': model.params[0],
                        'age_coef': model.params[1],
                        'age_se': model.bse[1],
                        'age_tstat': model.tvalues[1],
                        'age_pval': model.pvalues[1],
                        'sex_coef': model.params[2],
                        'sex_se': model.bse[2],
                        'sex_tstat': model.tvalues[2],
                        'sex_pval': model.pvalues[2],
                        'r_squared': model.rsquared,
                        'adj_r_squared': model.rsquared_adj,
                        'f_pval': model.f_pvalue
                    })
                else:
                    # Model: lr_means ~ age
                    X = data[['age']].values
                    y = data[lr_pair].values
                    X_with_const = sm.add_constant(X)
                    model = sm.OLS(y, X_with_const).fit()
                    
                    all_lm_results.append({
                        'expr_prop_threshold': threshold,
                        'source': source,
                        'target': target,
                        'LR_pair': lr_pair,
                        'n_samples': len(data),
                        'has_sex_variance': False,
                        'intercept': model.params[0],
                        'age_coef': model.params[1],
                        'age_se': model.bse[1],
                        'age_tstat': model.tvalues[1],
                        'age_pval': model.pvalues[1],
                        'sex_coef': np.nan,
                        'sex_se': np.nan,
                        'sex_tstat': np.nan,
                        'sex_pval': np.nan,
                        'r_squared': model.rsquared,
                        'adj_r_squared': model.rsquared_adj,
                        'f_pval': model.f_pvalue
                    })
            except:
                continue

# Convert to dataframe
lm_df = pd.DataFrame(all_lm_results)

# Save full results
lm_df.to_csv(output_dir / 'lm_results_all_thresholds.csv', index=False)
print(f"\n✓ Saved: {output_dir / 'lm_results_all_thresholds.csv'}")

print("\n" + "="*80)
print("LINEAR MODEL SUMMARY (lr_means ~ age + sex)")
print("="*80)

for threshold in thresholds:
    threshold_data = lm_df[lm_df['expr_prop_threshold'] == threshold]
    n_age_sig = (threshold_data['age_pval'] < 0.05).sum()
    n_sex_sig = (threshold_data['sex_pval'] < 0.05).sum()
    print(f"Threshold {threshold}: {len(threshold_data)} models, "
          f"Age sig (p<0.05): {n_age_sig}, Sex sig (p<0.05): {n_sex_sig}")

print("\nBy source-target pair:")
for threshold in thresholds:
    print(f"\n  Threshold {threshold}:")
    threshold_data = lm_df[lm_df['expr_prop_threshold'] == threshold]
    for source, target in target_pairs:
        pair_data = threshold_data[(threshold_data['source'] == source) & (threshold_data['target'] == target)]
        n_age_sig = (pair_data['age_pval'] < 0.05).sum()
        n_sex_sig = (pair_data['sex_pval'] < 0.05).sum()
        print(f"    {source}→{target}: {len(pair_data)} models, age sig: {n_age_sig}, sex sig: {n_sex_sig}")

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# 1. AGE P-VALUE DISTRIBUTIONS - FLEXIBLE Y-AXIS
print("Creating age p-value distributions (flexible y-axis)...")
fig, axes = plt.subplots(len(target_pairs), len(thresholds), figsize=(28, 16))

for row_idx, (source, target) in enumerate(target_pairs):
    for col_idx, threshold in enumerate(thresholds):
        ax = axes[row_idx, col_idx]
        
        pair_lm = lm_df[
            (lm_df['expr_prop_threshold'] == threshold) &
            (lm_df['source'] == source) &
            (lm_df['target'] == target)
        ]
        
        if len(pair_lm) > 0:
            pvals = pair_lm['age_pval'].values
            pvals_clean = pvals[~np.isnan(pvals)]
            
            if len(pvals_clean) > 0:
                ax.hist(pvals_clean, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                expected_height = len(pvals_clean) / 30
                ax.axhline(y=expected_height, color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.7)
                
                n_sig = np.sum(pvals_clean < 0.05)
                ax.text(0.98, 0.98, f'n={len(pair_lm)}\nsig={n_sig}', 
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=8, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        ax.set_title(f'{source}→{target}, expr_prop ≥ {threshold}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Age P-value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.tick_params(axis='both', labelsize=8)

plt.suptitle('Age Effect P-value Distributions (Flexible Y-axis)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'age_pvalue_distributions_flexible.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'age_pvalue_distributions_flexible.png'}")

# 2. AGE P-VALUE DISTRIBUTIONS - SHARED Y-AXIS
print("Creating age p-value distributions (shared y-axis)...")
fig, axes = plt.subplots(len(target_pairs), len(thresholds), figsize=(28, 16), sharey=True)

for row_idx, (source, target) in enumerate(target_pairs):
    for col_idx, threshold in enumerate(thresholds):
        ax = axes[row_idx, col_idx]
        
        pair_lm = lm_df[
            (lm_df['expr_prop_threshold'] == threshold) &
            (lm_df['source'] == source) &
            (lm_df['target'] == target)
        ]
        
        if len(pair_lm) > 0:
            pvals = pair_lm['age_pval'].values
            pvals_clean = pvals[~np.isnan(pvals)]
            
            if len(pvals_clean) > 0:
                ax.hist(pvals_clean, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                expected_height = len(pvals_clean) / 30
                ax.axhline(y=expected_height, color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.7)
                
                n_sig = np.sum(pvals_clean < 0.05)
                ax.text(0.98, 0.98, f'n={len(pair_lm)}\nsig={n_sig}', 
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=8, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        ax.set_title(f'{source}→{target}, expr_prop ≥ {threshold}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Age P-value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.tick_params(axis='both', labelsize=8)

plt.suptitle('Age Effect P-value Distributions (Shared Y-axis)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'age_pvalue_distributions_sharedy.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'age_pvalue_distributions_sharedy.png'}")

# 3. SEX P-VALUE DISTRIBUTIONS - FLEXIBLE Y-AXIS
print("Creating sex p-value distributions (flexible y-axis)...")
fig, axes = plt.subplots(len(target_pairs), len(thresholds), figsize=(28, 16))

for row_idx, (source, target) in enumerate(target_pairs):
    for col_idx, threshold in enumerate(thresholds):
        ax = axes[row_idx, col_idx]
        
        pair_lm = lm_df[
            (lm_df['expr_prop_threshold'] == threshold) &
            (lm_df['source'] == source) &
            (lm_df['target'] == target) &
            (lm_df['has_sex_variance'])
        ]
        
        if len(pair_lm) > 0:
            pvals = pair_lm['sex_pval'].values
            pvals_clean = pvals[~np.isnan(pvals)]
            
            if len(pvals_clean) > 0:
                ax.hist(pvals_clean, bins=30, edgecolor='black', alpha=0.7, color='coral')
                expected_height = len(pvals_clean) / 30
                ax.axhline(y=expected_height, color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.7)
                
                n_sig = np.sum(pvals_clean < 0.05)
                ax.text(0.98, 0.98, f'n={len(pair_lm)}\nsig={n_sig}', 
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=8, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        ax.set_title(f'{source}→{target}, expr_prop ≥ {threshold}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Sex P-value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.tick_params(axis='both', labelsize=8)

plt.suptitle('Sex Effect P-value Distributions (Flexible Y-axis)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'sex_pvalue_distributions_flexible.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'sex_pvalue_distributions_flexible.png'}")

# 4. SEX P-VALUE DISTRIBUTIONS - SHARED Y-AXIS
print("Creating sex p-value distributions (shared y-axis)...")
fig, axes = plt.subplots(len(target_pairs), len(thresholds), figsize=(28, 16), sharey=True)

for row_idx, (source, target) in enumerate(target_pairs):
    for col_idx, threshold in enumerate(thresholds):
        ax = axes[row_idx, col_idx]
        
        pair_lm = lm_df[
            (lm_df['expr_prop_threshold'] == threshold) &
            (lm_df['source'] == source) &
            (lm_df['target'] == target) &
            (lm_df['has_sex_variance'])
        ]
        
        if len(pair_lm) > 0:
            pvals = pair_lm['sex_pval'].values
            pvals_clean = pvals[~np.isnan(pvals)]
            
            if len(pvals_clean) > 0:
                ax.hist(pvals_clean, bins=30, edgecolor='black', alpha=0.7, color='coral')
                expected_height = len(pvals_clean) / 30
                ax.axhline(y=expected_height, color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.7)
                
                n_sig = np.sum(pvals_clean < 0.05)
                ax.text(0.98, 0.98, f'n={len(pair_lm)}\nsig={n_sig}', 
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=8, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        ax.set_title(f'{source}→{target}, expr_prop ≥ {threshold}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Sex P-value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.tick_params(axis='both', labelsize=8)

plt.suptitle('Sex Effect P-value Distributions (Shared Y-axis)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'sex_pvalue_distributions_sharedy.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'sex_pvalue_distributions_sharedy.png'}")

# 5. AGE QQ PLOTS
print("Creating age QQ plots...")
fig, axes = plt.subplots(len(target_pairs), len(thresholds), figsize=(28, 16))

for row_idx, (source, target) in enumerate(target_pairs):
    for col_idx, threshold in enumerate(thresholds):
        ax = axes[row_idx, col_idx]
        
        pair_lm = lm_df[
            (lm_df['expr_prop_threshold'] == threshold) &
            (lm_df['source'] == source) &
            (lm_df['target'] == target)
        ]
        
        if len(pair_lm) > 0:
            pvals = pair_lm['age_pval'].values
            pvals_clean = pvals[~np.isnan(pvals)]
            
            if len(pvals_clean) > 0:
                expected = np.linspace(0, 1, len(pvals_clean))
                observed = np.sort(pvals_clean)
                
                ax.scatter(expected, observed, alpha=0.5, s=10)
                ax.plot([0, 1], [0, 1], 'r--', linewidth=1, alpha=0.7)
                
                n_sig = np.sum(pvals_clean < 0.05)
                ax.text(0.98, 0.02, f'n={len(pair_lm)}\nsig={n_sig}', 
                       transform=ax.transAxes, ha='right', va='bottom',
                       fontsize=8, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        ax.set_title(f'{source}→{target}, expr_prop ≥ {threshold}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Expected P-value', fontsize=10)
        ax.set_ylabel('Observed P-value', fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.tick_params(axis='both', labelsize=8)

plt.suptitle('Age Effect QQ Plots', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'age_pvalue_qqplots.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'age_pvalue_qqplots.png'}")

# 6. SEX QQ PLOTS
print("Creating sex QQ plots...")
fig, axes = plt.subplots(len(target_pairs), len(thresholds), figsize=(28, 16))

for row_idx, (source, target) in enumerate(target_pairs):
    for col_idx, threshold in enumerate(thresholds):
        ax = axes[row_idx, col_idx]
        
        pair_lm = lm_df[
            (lm_df['expr_prop_threshold'] == threshold) &
            (lm_df['source'] == source) &
            (lm_df['target'] == target) &
            (lm_df['has_sex_variance'])
        ]
        
        if len(pair_lm) > 0:
            pvals = pair_lm['sex_pval'].values
            pvals_clean = pvals[~np.isnan(pvals)]
            
            if len(pvals_clean) > 0:
                expected = np.linspace(0, 1, len(pvals_clean))
                observed = np.sort(pvals_clean)
                
                ax.scatter(expected, observed, alpha=0.5, s=10, color='coral')
                ax.plot([0, 1], [0, 1], 'r--', linewidth=1, alpha=0.7)
                
                n_sig = np.sum(pvals_clean < 0.05)
                ax.text(0.98, 0.02, f'n={len(pair_lm)}\nsig={n_sig}', 
                       transform=ax.transAxes, ha='right', va='bottom',
                       fontsize=8, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        ax.set_title(f'{source}→{target}, expr_prop ≥ {threshold}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Expected P-value', fontsize=10)
        ax.set_ylabel('Observed P-value', fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.tick_params(axis='both', labelsize=8)

plt.suptitle('Sex Effect QQ Plots', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'sex_pvalue_qqplots.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'sex_pvalue_qqplots.png'}")

# 7. SUMMARY PLOTS
print("Creating summary metrics plot...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Number of LM models run
lm_counts = lm_df.groupby(['expr_prop_threshold', 'source', 'target']).size().reset_index(name='n_models')
for source, target in target_pairs:
    pair_counts = lm_counts[
        (lm_counts['source'] == source) & 
        (lm_counts['target'] == target)
    ]
    axes[0, 0].plot(pair_counts['expr_prop_threshold'], pair_counts['n_models'], 
                    marker='o', linewidth=2, markersize=8, label=f'{source}→{target}')

axes[0, 0].set_xlabel('expr_prop threshold', fontsize=11)
axes[0, 0].set_ylabel('Number of L-R pairs tested', fontsize=11)
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_title('Sample Size')

# Number of significant age effects
age_sig_counts = lm_df[lm_df['age_pval'] < 0.05].groupby(['expr_prop_threshold', 'source', 'target']).size().reset_index(name='n_sig')
for source, target in target_pairs:
    pair_sig = age_sig_counts[
        (age_sig_counts['source'] == source) & 
        (age_sig_counts['target'] == target)
    ]
    if len(pair_sig) > 0:
        axes[0, 1].plot(pair_sig['expr_prop_threshold'], pair_sig['n_sig'], 
                        marker='o', linewidth=2, markersize=8, label=f'{source}→{target}')

axes[0, 1].set_xlabel('expr_prop threshold', fontsize=11)
axes[0, 1].set_ylabel('Significant age effects (p<0.05)', fontsize=11)
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_title('Age Effects')

# Number of significant sex effects
sex_sig_counts = lm_df[lm_df['sex_pval'] < 0.05].groupby(['expr_prop_threshold', 'source', 'target']).size().reset_index(name='n_sig')
for source, target in target_pairs:
    pair_sig = sex_sig_counts[
        (sex_sig_counts['source'] == source) & 
        (sex_sig_counts['target'] == target)
    ]
    if len(pair_sig) > 0:
        axes[0, 2].plot(pair_sig['expr_prop_threshold'], pair_sig['n_sig'], 
                        marker='o', linewidth=2, markersize=8, label=f'{source}→{target}')

axes[0, 2].set_xlabel('expr_prop threshold', fontsize=11)
axes[0, 2].set_ylabel('Significant sex effects (p<0.05)', fontsize=11)
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_title('Sex Effects')

# Mean samples per LM
mean_samples = lm_df.groupby(['expr_prop_threshold', 'source', 'target'])['n_samples'].mean().reset_index()
for source, target in target_pairs:
    pair_samples = mean_samples[
        (mean_samples['source'] == source) & 
        (mean_samples['target'] == target)
    ]
    axes[1, 0].plot(pair_samples['expr_prop_threshold'], pair_samples['n_samples'], 
                    marker='o', linewidth=2, markersize=8, label=f'{source}→{target}')

axes[1, 0].set_xlabel('expr_prop threshold', fontsize=11)
axes[1, 0].set_ylabel('Mean individuals per L-R pair', fontsize=11)
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_title('Data Availability')

# Proportion of LMs with sig age effect
prop_age_sig = []
for threshold in thresholds:
    for source, target in target_pairs:
        threshold_pair = lm_df[
            (lm_df['expr_prop_threshold'] == threshold) &
            (lm_df['source'] == source) &
            (lm_df['target'] == target)
        ]
        if len(threshold_pair) > 0:
            prop_sig = (threshold_pair['age_pval'] < 0.05).sum() / len(threshold_pair)
            prop_age_sig.append({
                'threshold': threshold,
                'pair': f'{source}→{target}',
                'prop_sig': prop_sig
            })

prop_df = pd.DataFrame(prop_age_sig)
for pair in prop_df['pair'].unique():
    pair_data = prop_df[prop_df['pair'] == pair]
    axes[1, 1].plot(pair_data['threshold'], pair_data['prop_sig'], 
                    marker='o', linewidth=2, markersize=8, label=pair)

axes[1, 1].set_xlabel('expr_prop threshold', fontsize=11)
axes[1, 1].set_ylabel('Proportion with sig age effect', fontsize=11)
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_title('Age Effect Rate')

# Proportion of LMs with sig sex effect
prop_sex_sig = []
for threshold in thresholds:
    for source, target in target_pairs:
        threshold_pair = lm_df[
            (lm_df['expr_prop_threshold'] == threshold) &
            (lm_df['source'] == source) &
            (lm_df['target'] == target) &
            (lm_df['has_sex_variance'])
        ]
        if len(threshold_pair) > 0:
            prop_sig = (threshold_pair['sex_pval'] < 0.05).sum() / len(threshold_pair)
            prop_sex_sig.append({
                'threshold': threshold,
                'pair': f'{source}→{target}',
                'prop_sig': prop_sig
            })

prop_sex_df = pd.DataFrame(prop_sex_sig)
for pair in prop_sex_df['pair'].unique():
    pair_data = prop_sex_df[prop_sex_df['pair'] == pair]
    axes[1, 2].plot(pair_data['threshold'], pair_data['prop_sig'], 
                    marker='o', linewidth=2, markersize=8, label=pair)

axes[1, 2].set_xlabel('expr_prop threshold', fontsize=11)
axes[1, 2].set_ylabel('Proportion with sig sex effect', fontsize=11)
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_title('Sex Effect Rate')

plt.tight_layout()
plt.savefig(output_dir / 'summary_metrics.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'summary_metrics.png'}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Results in: {output_dir}")
print(f"\nFiles created:")
print(f"  - lm_results_all_thresholds.csv")
print(f"  - age_pvalue_distributions_flexible.png")
print(f"  - age_pvalue_distributions_sharedy.png")
print(f"  - sex_pvalue_distributions_flexible.png")
print(f"  - sex_pvalue_distributions_sharedy.png")
print(f"  - age_pvalue_qqplots.png")
print(f"  - sex_pvalue_qqplots.png")
print(f"  - summary_metrics.png")
