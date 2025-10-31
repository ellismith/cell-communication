#!/usr/bin/env python3
"""
Analyze age and sex effects on cell-cell communication
"""
import scanpy as sc
import pandas as pd
import numpy as np
import liana as li

print("="*70)
print("Analyzing Age and Sex Effects on Communication")
print("="*70)

# Load data
print("\n[1] Loading data...")
astro_path = "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/adult_grn_astrocytes_selected.h5ad"
gaba_path = "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/adult_grn_gaba_selected.h5ad"

adata_astro = sc.read_h5ad(astro_path)
adata_gaba = sc.read_h5ad(gaba_path)

adata_astro.obs['cell_type'] = 'Astrocyte'
adata_gaba.obs['cell_type'] = 'GABA_neuron'

# Combine
adata_combined = sc.concat([adata_astro, adata_gaba], join='inner')

print(f"  Total cells: {adata_combined.n_obs:,}")
print(f"  Age range: {adata_combined.obs['age'].min():.2f} - {adata_combined.obs['age'].max():.2f}")

# Create age groups
print("\n[2] Creating age groups...")
adata_combined.obs['age_group'] = pd.cut(
    adata_combined.obs['age'],
    bins=[0, 8, 15, 25],
    labels=['Young', 'Middle', 'Old']
)

print(adata_combined.obs.groupby(['age_group', 'cell_type']).size())

# Analyze by age group
print("\n[3] Running CellChat by age group...")

age_results = {}
for age_group in ['Young', 'Middle', 'Old']:
    print(f"\n  Processing {age_group} group...")
    
    adata_age = adata_combined[adata_combined.obs['age_group'] == age_group].copy()
    
    # Normalize
    sc.pp.normalize_total(adata_age, target_sum=1e4)
    sc.pp.log1p(adata_age)
    
    # Run liana
    li.mt.rank_aggregate(
        adata_age,
        groupby='cell_type',
        resource_name='cellchatdb',
        expr_prop=0.1,
        use_raw=False,
        n_perms=50,
        verbose=False
    )
    
    age_results[age_group] = adata_age.uns['liana_res'].copy()
    age_results[age_group]['age_group'] = age_group
    
    n_sig = (age_results[age_group]['magnitude_rank'] < 0.05).sum()
    print(f"    Found {n_sig} significant interactions")

# Combine results
all_age_results = pd.concat(age_results.values(), ignore_index=True)
all_age_results.to_csv('/scratch/easmit31/cell_cell/results/cellchat_by_age.csv', index=False)
print("\n✓ Saved age-stratified results")

# Analyze by sex
print("\n[4] Running CellChat by sex...")

sex_results = {}
for sex in ['F', 'M']:
    print(f"\n  Processing {sex}...")
    
    adata_sex = adata_combined[adata_combined.obs['sex'] == sex].copy()
    
    # Normalize
    sc.pp.normalize_total(adata_sex, target_sum=1e4)
    sc.pp.log1p(adata_sex)
    
    # Run liana
    li.mt.rank_aggregate(
        adata_sex,
        groupby='cell_type',
        resource_name='cellchatdb',
        expr_prop=0.1,
        use_raw=False,
        n_perms=50,
        verbose=False
    )
    
    sex_results[sex] = adata_sex.uns['liana_res'].copy()
    sex_results[sex]['sex'] = sex
    
    n_sig = (sex_results[sex]['magnitude_rank'] < 0.05).sum()
    print(f"    Found {n_sig} significant interactions")

# Combine sex results
all_sex_results = pd.concat(sex_results.values(), ignore_index=True)
all_sex_results.to_csv('/scratch/easmit31/cell_cell/results/cellchat_by_sex.csv', index=False)
print("\n✓ Saved sex-stratified results")

print("\n" + "="*70)
print("Age/Sex Analysis Complete!")
print("="*70)
