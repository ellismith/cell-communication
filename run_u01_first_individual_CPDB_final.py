#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import numpy as np
import liana as li
from statsmodels.stats.multitest import multipletests
import os

individual = "0B9"
base_path = "/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/"
output_dir = "/scratch/easmit31/cell_cell/results/u01_per_individual"
os.makedirs(output_dir, exist_ok=True)

cell_types = {
    'Astrocyte': 'Res1_astrocytes_update.h5ad',
    'Basket': 'Res1_basket-cells_update.h5ad',
    'Cerebellar': 'Res1_cerebellar-neurons_subset.h5ad',
    'Ependymal': 'Res1_ependymal-cells_new.h5ad',
    'GABA': 'Res1_GABAergic-neurons_subset.h5ad',
    'Glutamatergic': 'Res1_glutamatergic-neurons_update.h5ad',
    'MSN': 'Res1_medium-spiny-neurons_subset.h5ad',
    'Microglia': 'Res1_microglia_new.h5ad',
    'Midbrain': 'Res1_midbrain-neurons_update.h5ad',
    'OPC_Oligo': 'Res1_opc-olig_subset.h5ad',
    'Vascular': 'Res1_vascular-cells_subset.h5ad'
}

print(f"Loading data for individual {individual}...")
adatas = []
for cell_type, filename in cell_types.items():
    adata = sc.read_h5ad(base_path + filename)
    adata = adata[adata.obs['animal_id'] == individual].copy()
    if adata.n_obs == 0:
        continue
    
    has_name = ~adata.var['external_gene_name'].isna()
    adata = adata[:, has_name].copy()
    adata.var_names = adata.var['external_gene_name'].astype(str).values
    adata.var_names_make_unique()
    
    adata.obs["cell_type"] = cell_type
    adata.obs["cell_type_region"] = adata.obs["cell_type"] + "_" + adata.obs["region"].astype(str)
    adatas.append(adata)

adata_combined = sc.concat(adatas, join="inner")
print(f"Combined: {adata_combined.n_obs:,} cells, {adata_combined.n_vars:,} genes")

# Normalize
adata_combined.X[pd.isna(adata_combined.X)] = 0
sc.pp.normalize_total(adata_combined, target_sum=1e4)
sc.pp.log1p(adata_combined)

# Load interactions as list of tuples (gene_a, gene_b)
print("\nLoading CellPhoneDB interactions (protein-protein only, no complexes)...")
interactions_df = pd.read_csv("/scratch/easmit31/cell_cell/cellphonedb_interactions_gene_symbols.csv")
interactions_list = list(zip(interactions_df['gene_a'], interactions_df['gene_b']))
print(f"Loaded {len(interactions_list)} interactions")

# Run LIANA
print(f"\nRunning LIANA rank_aggregate for individual {individual}...")
li.mt.rank_aggregate(
    adata_combined,
    groupby="cell_type_region",
    interactions=interactions_list,
    expr_prop=0.1,
    use_raw=False,
    n_perms=100,
    verbose=True
)

# Save results with FDR
print("\nSaving results...")
res = adata_combined.uns["liana_res"].copy()
_, res['qval'], _, _ = multipletests(res['magnitude_rank'], method='fdr_bh')

res['animal_id'] = individual
res['age'] = adata_combined.obs['age'].iloc[0]
res['sex'] = adata_combined.obs['sex'].iloc[0]

out_path = os.path.join(output_dir, f"{individual}_results.csv")
res.to_csv(out_path, index=False)

sig = res[res["qval"] < 0.05]
print(f"\n✓ Saved {out_path}")
print(f"  Individual: {individual}")
print(f"  Age: {res['age'].iloc[0]}, Sex: {res['sex'].iloc[0]}")
print(f"  Total interactions tested: {len(res):,}")
print(f"  Significant (FDR q<0.05): {len(sig):,}")
