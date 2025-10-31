#!/usr/bin/env python3
import scanpy as sc
import pandas as pd

print("Checking GABA regions in detail...")

gaba_path = "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/adult_grn_gaba_aucell.h5ad"
adata_gaba = sc.read_h5ad(gaba_path)

print(f"\nTotal GABA cells: {adata_gaba.n_obs:,}")

print("\nChecking 'region' column:")
print(adata_gaba.obs['region'].value_counts())

print("\nUnique regions:")
print(sorted(adata_gaba.obs['region'].unique()))

print("\nChecking other potential region columns:")
for col in adata_gaba.obs.columns:
    if 'region' in col.lower() or 'area' in col.lower() or 'brain' in col.lower():
        print(f"\n{col}:")
        if adata_gaba.obs[col].nunique() < 50:
            print(adata_gaba.obs[col].value_counts().head(20))
        else:
            print(f"  Too many unique values: {adata_gaba.obs[col].nunique()}")

