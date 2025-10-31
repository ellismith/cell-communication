#!/usr/bin/env python3
"""
ALL 6 cell types communicating - FULL DATA, NO DOWNSAMPLING
"""
import scanpy as sc
import pandas as pd
import liana as li
import os

print("="*70)
print("ALL vs ALL Cell Type Communication - 6 Cell Types, Full Data")
print("="*70)

base_path = "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/"

# ALL 6 cell types (no basket)
cell_types = {
    'Astrocyte': 'adult_grn_astrocytes_selected.h5ad',
    'GABA': 'adult_grn_gaba_selected.h5ad',
    'Glutamatergic': 'adult_grn_glut_selected.h5ad',
    'MSN': 'adult_grn_MSN_selected.h5ad',
    'OPC_Oligo': 'adult_grn_opc-olig_selected.h5ad',
    'Microglia': 'adult_grn_microglia_selected.h5ad'
}

# Load ALL - NO DOWNSAMPLING
print("\nLoading ALL cell types (full dataset)...")
adatas = []
total_cells = 0
for name, file in cell_types.items():
    print(f"  Loading {name}...")
    adata = sc.read_h5ad(base_path + file)
    adata.obs['cell_type'] = name
    adatas.append(adata)
    print(f"    {adata.n_obs:,} cells")
    total_cells += adata.n_obs

print(f"\nTotal cells to process: {total_cells:,}")

print("\nCombining all cell types...")
adata_all = sc.concat(adatas, join='inner')
print(f"Combined: {adata_all.n_obs:,} cells x {adata_all.n_vars:,} genes")
print("\nCell type distribution:")
print(adata_all.obs['cell_type'].value_counts())

print("\nNormalizing...")
sc.pp.normalize_total(adata_all, target_sum=1e4)
sc.pp.log1p(adata_all)

print("\nRunning CellChat (this will take a while)...")
li.mt.rank_aggregate(
    adata_all,
    groupby='cell_type',
    resource_name='cellchatdb',
    expr_prop=0.1,
    use_raw=False,
    n_perms=100,
    verbose=True
)

print("\n✓ CellChat complete!")

# Save results
results = adata_all.uns['liana_res']
output_dir = "/scratch/easmit31/cell_cell/results"
os.makedirs(output_dir, exist_ok=True)

results.to_csv(f"{output_dir}/cellchat_all_6celltypes_full.csv", index=False)

sig = results[results['magnitude_rank'] < 0.05]
sig.to_csv(f"{output_dir}/cellchat_all_6celltypes_significant.csv", index=False)

print(f"\n✓ Total interactions tested: {len(results):,}")
print(f"✓ Significant interactions: {len(sig):,}")

# Summary by cell type pair
print("\n" + "="*70)
print("SIGNIFICANT INTERACTIONS BY CELL TYPE PAIR")
print("="*70)
for source in sorted(sig['source'].unique()):
    for target in sorted(sig['target'].unique()):
        if source != target:
            pair_data = sig[(sig['source'] == source) & (sig['target'] == target)]
            if len(pair_data) > 0:
                print(f"{source} → {target}: {len(pair_data)} interactions")

print("\n" + "="*70)
print("DONE!")
print("="*70)
