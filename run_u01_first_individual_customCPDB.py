#!/usr/bin/env python3
"""
Run LIANA pairwise analysis for the first individual using a custom CellPhoneDB CSV resource
"""
import os
import scanpy as sc
import pandas as pd
import numpy as np
import liana as li
from statsmodels.stats.multitest import multipletests

base_path = "/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/"
output_dir = "/scratch/easmit31/cell_cell/results/u01_first_individual_customCPDB"
os.makedirs(output_dir, exist_ok=True)

expected_counts_path = "/scratch/easmit31/cell_cell/u01_cell_counts_per_individual.csv"
expected_counts = pd.read_csv(expected_counts_path)
first_individual = sorted(expected_counts['animal_id'].unique())[0]
print(f"Running analysis for individual: {first_individual}")

# Paths to your downloaded CellPhoneDB CSVs
cpdb_path = "/scratch/easmit31/cell_cell/CellPhoneDB_latest/"
interaction_csv = os.path.join(cpdb_path, "interaction_input_CellPhoneDB.csv")
complex_csv = os.path.join(cpdb_path, "complex_input_CellPhoneDB.csv")
cofactor_csv = os.path.join(cpdb_path, "cofactor_input_CellPhoneDB.csv")
geneInfo_csv = os.path.join(cpdb_path, "geneInfo_input_CellPhoneDB.csv")

# Load CSVs into a LIANA resource dictionary
custom_resource = {
    "interaction": pd.read_csv(interaction_csv),
    "complex": pd.read_csv(complex_csv),
    "cofactor": pd.read_csv(cofactor_csv),
    "geneInfo": pd.read_csv(geneInfo_csv)
}

# Define cell types
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

adatas = []
for ct, filename in cell_types.items():
    adata = sc.read_h5ad(os.path.join(base_path, filename))
    adata = adata[adata.obs['animal_id'] == first_individual].copy()
    if adata.n_obs == 0:
        continue
    has_name = ~adata.var['external_gene_name'].isna()
    adata = adata[:, has_name].copy()
    adata.var_names = adata.var['external_gene_name'].values
    adata.var_names_make_unique()
    adata.obs['cell_type'] = ct
    adata.obs['cell_type_region'] = adata.obs['cell_type'] + "_" + adata.obs['region'].astype(str)
    adatas.append(adata)

if len(adatas) == 0:
    print(f"No data for individual {first_individual}, exiting")
    exit(0)

adata_combined = sc.concat(adatas, join="inner")
adata_combined.X[pd.isna(adata_combined.X)] = 0
sc.pp.normalize_total(adata_combined, target_sum=1e4)
sc.pp.log1p(adata_combined)

print("\nRunning LIANA with custom CellPhoneDB resource...")
li.mt.rank_aggregate(
    adata_combined,
    groupby="cell_type_region",
    resource_name=custom_resource,
    expr_prop=0.1,
    use_raw=False,
    n_perms=100,
    verbose=True
)

res = adata_combined.uns["liana_res"].copy()
_, res['qval'], _, _ = multipletests(res['magnitude_rank'], method='fdr_bh')
res['animal_id'] = first_individual
res['age'] = adata_combined.obs['age'].iloc[0]
res['sex'] = adata_combined.obs['sex'].iloc[0]

out_path = os.path.join(output_dir, f"{first_individual}_results.csv")
res.to_csv(out_path, index=False)
print(f"\n✓ Saved results for {first_individual} at {out_path}")
