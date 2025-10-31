#!/usr/bin/env python3
"""
Comprehensive CellChat analysis: Astrocytes communicating with all cell types
"""
import scanpy as sc
import pandas as pd
import numpy as np
import liana as li
import matplotlib.pyplot as plt
import os

print("="*70)
print("Comprehensive Multi-Cell Type Communication Analysis")
print("="*70)

# Define all cell types to load
cell_types = {
    'Astrocyte': 'adult_grn_astrocytes_selected.h5ad',
    'GABA': 'adult_grn_gaba_selected.h5ad',
    'Glutamatergic': 'adult_grn_glut_selected.h5ad',
    'MSN': 'adult_grn_MSN_selected.h5ad',
    'OPC_Oligo': 'adult_grn_opc-olig_selected.h5ad',
    'Basket': 'adult_grn_basket_selected.h5ad',
    'Microglia': 'adult_grn_microglia_selected.h5ad'
}

base_path = "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/"

# Load all datasets
print("\n[Step 1] Loading all cell types...")
adatas = {}
for cell_type, filename in cell_types.items():
    print(f"  Loading {cell_type}...")
    adata = sc.read_h5ad(base_path + filename)
    adata.obs['cell_type'] = cell_type
    adatas[cell_type] = adata
    print(f"    {adata.n_obs:,} cells")

# Combine all
print("\n[Step 2] Combining datasets...")
adata_combined = sc.concat(list(adatas.values()), join='inner')

print(f"\n  Combined: {adata_combined.n_obs:,} cells x {adata_combined.n_vars:,} genes")
print(f"\n  Cell type distribution:")
print(adata_combined.obs['cell_type'].value_counts())

# Normalize
print("\n[Step 3] Normalizing data...")
sc.pp.normalize_total(adata_combined, target_sum=1e4)
sc.pp.log1p(adata_combined)
print("  ✓ Data normalized")

# Run CellChat
print("\n[Step 4] Running comprehensive cell-cell communication analysis...")
print("  This will take 30-60 minutes for all cell type pairs...")

li.mt.rank_aggregate(
    adata_combined,
    groupby='cell_type',
    resource_name='cellchatdb',
    expr_prop=0.1,
    use_raw=False,
    n_perms=100,
    verbose=True
)

print("\n  ✓ Communication analysis complete!")

# Save results
comm_results = adata_combined.uns['liana_res']
print(f"\n  Total interactions tested: {len(comm_results):,}")

output_dir = "/scratch/easmit31/cell_cell/results"
os.makedirs(output_dir, exist_ok=True)

comm_results.to_csv(f"{output_dir}/cellchat_all_celltypes.csv", index=False)

# Filter significant
significant = comm_results[comm_results['magnitude_rank'] < 0.05].copy()
significant = significant.sort_values('magnitude_rank')
print(f"  Significant interactions: {len(significant)}")

significant.to_csv(f"{output_dir}/cellchat_all_celltypes_significant.csv", index=False)

# Get astrocyte-specific interactions
astro_interactions = significant[
    (significant['source'] == 'Astrocyte') | (significant['target'] == 'Astrocyte')
].copy()

print(f"\n  Astrocyte-involved interactions: {len(astro_interactions)}")
astro_interactions.to_csv(f"{output_dir}/cellchat_astrocyte_interactions.csv", index=False)

# Summary by cell type pair
print("\n" + "="*70)
print("ASTROCYTE COMMUNICATION SUMMARY")
print("="*70)

astro_out = astro_interactions[astro_interactions['source'] == 'Astrocyte']
astro_in = astro_interactions[astro_interactions['target'] == 'Astrocyte']

print("\nAstrocyte OUTGOING (Astrocyte → other cell types):")
for target in astro_out['target'].unique():
    count = len(astro_out[astro_out['target'] == target])
    print(f"  → {target}: {count} interactions")

print("\nAstrocyte INCOMING (other cell types → Astrocyte):")
for source in astro_in['source'].unique():
    count = len(astro_in[astro_in['source'] == source])
    print(f"  {source} →: {count} interactions")

print("\n" + "="*70)
print("Analysis Complete!")
print("="*70)

