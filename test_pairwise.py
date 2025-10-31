#!/usr/bin/env python3
"""
Test script - just Microglia ↔ OPC_Oligo (smaller files)
"""
import scanpy as sc
import pandas as pd
import liana as li
import os

print("="*70)
print("TEST: Analyzing Microglia ↔ OPC_Oligo")
print("="*70)

base_path = "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/"

# --- Load both cell types
adata1 = sc.read_h5ad(base_path + "adult_grn_microglia_selected_converted.h5ad")
adata2 = sc.read_h5ad(base_path + "adult_grn_opc-olig_selected_converted.h5ad")

print(f"Loaded: {adata1.n_obs:,} Microglia cells, {adata2.n_obs:,} OPC_Oligo cells")
print(f"Regions in data: {adata1.obs['region'].unique()}")

# --- Tag cell types
adata1.obs["cell_type"] = "Microglia"
adata2.obs["cell_type"] = "OPC_Oligo"

# --- Create combined grouping: cell_type + region
adata1.obs["cell_type_region"] = adata1.obs["cell_type"] + "_" + adata1.obs["region"].astype(str)
adata2.obs["cell_type_region"] = adata2.obs["cell_type"] + "_" + adata2.obs["region"].astype(str)

print(f"Cell type_region groups: {sorted(adata1.obs['cell_type_region'].unique())}")

# --- Combine
adata_pair = sc.concat([adata1, adata2], join="inner")
print(f"Combined: {adata_pair.n_obs:,} cells, {adata_pair.n_vars:,} genes")

# --- Fill NaNs and normalize
adata_pair.X[pd.isna(adata_pair.X)] = 0
sc.pp.normalize_total(adata_pair, target_sum=1e4)
sc.pp.log1p(adata_pair)

# --- Run LIANA
print("\nRunning CellChat with region preserved...")
li.mt.rank_aggregate(
    adata_pair,
    groupby="cell_type_region",
    resource_name="cellchatdb",
    expr_prop=0.1,
    use_raw=False,
    n_perms=100,
    verbose=True
)

# --- Check results
res = adata_pair.uns["liana_res"].copy()
print(f"\nResults shape: {res.shape}")
print(f"\nSource groups: {sorted(res['source'].unique())}")
print(f"Target groups: {sorted(res['target'].unique())}")
print(f"\nFirst few rows:")
print(res.head(10))

# --- Save
res.to_csv("test_microglia_opc_results.csv", index=False)
print("\n✓ Saved test_microglia_opc_results.csv")
