#!/usr/bin/env python3
"""
Test script to load and examine astrocyte + GABA data for CellChat
"""
import scanpy as sc
import pandas as pd
import numpy as np

print("="*60)
print("Loading Adult Astrocyte and GABA Data")
print("="*60)

# Load the data
print("\nLoading astrocytes...")
astro_path = "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/adult_grn_astrocytes_selected.h5ad"
adata_astro = sc.read_h5ad(astro_path)

print(f"Astrocytes: {adata_astro.n_obs:,} cells x {adata_astro.n_vars:,} genes")

print("\nLoading GABAergic neurons...")
gaba_path = "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/adult_grn_gaba_selected.h5ad"
adata_gaba = sc.read_h5ad(gaba_path)

print(f"GABA neurons: {adata_gaba.n_obs:,} cells x {adata_gaba.n_vars:,} genes")

# Combined totals
print("\n" + "="*60)
print("Combined Dataset Stats")
print("="*60)
total_cells = adata_astro.n_obs + adata_gaba.n_obs
print(f"Total cells: {total_cells:,}")
print(f"  - Astrocytes: {adata_astro.n_obs:,} ({100*adata_astro.n_obs/total_cells:.1f}%)")
print(f"  - GABA neurons: {adata_gaba.n_obs:,} ({100*adata_gaba.n_obs/total_cells:.1f}%)")

# Check gene overlap
astro_genes = set(adata_astro.var_names)
gaba_genes = set(adata_gaba.var_names)
shared_genes = astro_genes & gaba_genes

print(f"\nGene counts:")
print(f"  - Astrocytes: {len(astro_genes):,} genes")
print(f"  - GABA neurons: {len(gaba_genes):,} genes")
print(f"  - Shared genes: {len(shared_genes):,} genes")
print(f"  - Astrocyte-only: {len(astro_genes - gaba_genes):,} genes")
print(f"  - GABA-only: {len(gaba_genes - astro_genes):,} genes")

# Check metadata
print("\n" + "="*60)
print("Metadata Available")
print("="*60)

print("\nAstrocyte metadata columns:")
print(adata_astro.obs.columns.tolist())

print("\nGABA metadata columns:")
print(adata_gaba.obs.columns.tolist())

# Check for key variables
if 'age' in adata_astro.obs.columns:
    print(f"\n--- AGE DISTRIBUTION ---")
    print(f"Astrocytes: {adata_astro.obs['age'].min():.2f} - {adata_astro.obs['age'].max():.2f} years")
    print(f"  Mean: {adata_astro.obs['age'].mean():.2f}, Median: {adata_astro.obs['age'].median():.2f}")
    print(f"GABA: {adata_gaba.obs['age'].min():.2f} - {adata_gaba.obs['age'].max():.2f} years")
    print(f"  Mean: {adata_gaba.obs['age'].mean():.2f}, Median: {adata_gaba.obs['age'].median():.2f}")

if 'sex' in adata_astro.obs.columns:
    print(f"\n--- SEX DISTRIBUTION ---")
    print(f"Astrocytes:")
    astro_sex = adata_astro.obs['sex'].value_counts()
    for sex, count in astro_sex.items():
        print(f"  {sex}: {count:,} ({100*count/len(adata_astro):.1f}%)")
    
    print(f"GABA:")
    gaba_sex = adata_gaba.obs['sex'].value_counts()
    for sex, count in gaba_sex.items():
        print(f"  {sex}: {count:,} ({100*count/len(adata_gaba):.1f}%)")

if 'region' in adata_astro.obs.columns:
    print(f"\n--- BRAIN REGIONS ---")
    print(f"Astrocytes: {adata_astro.obs['region'].nunique()} regions")
    print("Top 10 regions:")
    for region, count in adata_astro.obs['region'].value_counts().head(10).items():
        print(f"  {region}: {count:,} cells")
    
    print(f"\nGABA: {adata_gaba.obs['region'].nunique()} regions")
    print("Top 10 regions:")
    for region, count in adata_gaba.obs['region'].value_counts().head(10).items():
        print(f"  {region}: {count:,} cells")
    
    # Check region overlap
    astro_regions = set(adata_astro.obs['region'].unique())
    gaba_regions = set(adata_gaba.obs['region'].unique())
    shared_regions = astro_regions & gaba_regions
    print(f"\nShared regions: {len(shared_regions)} (both cell types present)")

# Check expression data format
print("\n" + "="*60)
print("Expression Data Format")
print("="*60)
print(f"\nAstrocytes X matrix type: {type(adata_astro.X)}")
print(f"GABA X matrix type: {type(adata_gaba.X)}")

# Check if data is normalized
if hasattr(adata_astro.X, 'max'):
    max_val_astro = adata_astro.X.max() if hasattr(adata_astro.X, 'max') else adata_astro.X.data.max()
    print(f"Max expression value (astrocytes): {max_val_astro:.2f}")
    
    max_val_gaba = adata_gaba.X.max() if hasattr(adata_gaba.X, 'max') else adata_gaba.X.data.max()
    print(f"Max expression value (GABA): {max_val_gaba:.2f}")
    
    if max_val_astro > 100 or max_val_gaba > 100:
        print("⚠️  Data appears to be raw counts (not normalized)")
    else:
        print("✓ Data appears to be log-normalized")

# Check for sparsity
from scipy.sparse import issparse
if issparse(adata_astro.X):
    sparsity_astro = 1 - (adata_astro.X.nnz / (adata_astro.n_obs * adata_astro.n_vars))
    print(f"\nAstrocyte matrix sparsity: {100*sparsity_astro:.1f}% zeros")
if issparse(adata_gaba.X):
    sparsity_gaba = 1 - (adata_gaba.X.nnz / (adata_gaba.n_obs * adata_gaba.n_vars))
    print(f"GABA matrix sparsity: {100*sparsity_gaba:.1f}% zeros")

print("\n" + "="*60)
print("Data Quality Checks")
print("="*60)

# Check for any NaN or inf values
print(f"\nAstrocytes - NaN values: {pd.isna(adata_astro.X.data if issparse(adata_astro.X) else adata_astro.X).sum()}")
print(f"GABA - NaN values: {pd.isna(adata_gaba.X.data if issparse(adata_gaba.X) else adata_gaba.X).sum()}")

print("\n" + "="*60)
print("Ready for CellChat Analysis!")
print("="*60)
print("\nNext steps:")
print("1. Combine astrocyte + GABA datasets")
print("2. Ensure proper normalization for CellChat")
print("3. Run CellChat to identify communication patterns")

