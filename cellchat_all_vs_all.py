#!/usr/bin/env python3
"""
ALL cell types communicating with ALL cell types
"""
import scanpy as sc
import pandas as pd
import numpy as np
import liana as li
import os

print("="*70)
print("ALL vs ALL Cell Type Communication Analysis")
print("="*70)

base_path = "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/"

# ALL 7 cell types
cell_types = {
    'Astrocyte': 'adult_grn_astrocytes_selected.h5ad',
    'GABA': 'adult_grn_gaba_selected.h5ad',
    'Glutamatergic': 'adult_grn_glut_selected.h5ad',
    'MSN': 'adult_grn_MSN_selected.h5ad',
    'OPC_Oligo': 'adult_grn_opc-olig_selected.h5ad',
    'Basket': 'adult_grn_basket_selected.h5ad',
    'Microglia': 'adult_grn_microglia_selected.h5ad'
}

# Load ALL
print("\nLoading ALL cell types...")
adatas = []
for name, file in cell_types.items():
    print(f"  {name}...")
    adata = sc.read_h5ad(base_path + file)
    
    # Downsample big ones to 50K max
    if adata.n_obs > 50000:
        np.random.seed(42)
        idx = np.random.choice(adata.n_obs, 50000, replace=False)
        adata = adata[idx].copy()
    
    adata.obs['cell_type'] = name
    adatas.append(adata)
    print(f"    {adata.n_obs:,} cells")

# COMBINE ALL
print("\nCombining ALL...")
adata_all = sc.concat(adatas, join='inner')
print(f"Total: {adata_all.n_obs:,} cells")
print(adata_all.obs['cell_type'].value_counts())

# Normalize
print("\nNormalizing...")
sc.pp.normalize_total(adata_all, target_sum=1e4)
sc.pp.log1p(adata_all)

# RUN CELLCHAT - ALL vs ALL
print("\nRunning CellChat (ALL cell types)...")
li.mt.rank_aggregate(
    adata_all,
    groupby='cell_type',
    resource_name='cellchatdb',
    expr_prop=0.1,
    use_raw=False,
    n_perms=50,
    verbose=True
)

# Save
results = adata_all.uns['liana_res']
output_dir = "/scratch/easmit31/cell_cell/results"
os.makedirs(output_dir, exist_ok=True)

results.to_csv(f"{output_dir}/cellchat_all_vs_all.csv", index=False)

sig = results[results['magnitude_rank'] < 0.05]
sig.to_csv(f"{output_dir}/cellchat_all_vs_all_significant.csv", index=False)

print(f"\n✓ Total: {len(results):,}")
print(f"✓ Significant: {len(sig):,}")
print("\nDONE!")
