#!/usr/bin/env python3
import scanpy as sc
import os

print("Investigating GABA files...")

# Check what GABA files exist
gaba_dir = "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/"
print(f"\nFiles in {gaba_dir}:")
os.system(f"ls -lh {gaba_dir}/*gaba* 2>/dev/null")

# Also check the other location you mentioned
other_dir = "/scratch/easmit31/GRN_copy/scenic/h5ad_files/"
print(f"\nFiles in {other_dir}:")
os.system(f"ls -lh {other_dir}/*gaba*adult* 2>/dev/null")

# Load the current file to see chunk info
current_path = "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/adult_grn_gaba_aucell.h5ad"
adata = sc.read_h5ad(current_path)

print(f"\n\nCurrent file ({current_path}):")
print(f"  Cells: {adata.n_obs:,}")
print(f"  Genes: {adata.n_vars:,}")
print(f"  Regions: {adata.obs['region'].unique()}")

# Check if there's chunk info in the metadata
print("\n\nChecking for chunk information in obs columns:")
chunk_cols = [col for col in adata.obs.columns if 'chunk' in col.lower()]
if chunk_cols:
    print(f"Found chunk columns: {chunk_cols}")
    for col in chunk_cols:
        print(f"  {col}: {adata.obs[col].unique()}")
else:
    print("  No chunk columns found")

# Check the full dataset location
print("\n\nChecking for full GABA dataset...")
full_paths = [
    "/scratch/easmit31/GRN_copy/scenic/h5ad_files/gaba_adults_allLv_pyscenic_output_merged.h5ad",
    "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/adult_grn_gaba_fullset_liftover.h5ad"
]

for path in full_paths:
    if os.path.exists(path):
        print(f"\nFound: {path}")
        try:
            adata_full = sc.read_h5ad(path)
            print(f"  Cells: {adata_full.n_obs:,}")
            print(f"  Regions: {adata_full.obs['region'].nunique()} unique regions")
            print(f"  Region counts:")
            print(adata_full.obs['region'].value_counts().head(10))
        except Exception as e:
            print(f"  Error loading: {e}")

