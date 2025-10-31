#!/usr/bin/env python3
import scanpy as sc

print("Checking gene names in the data...")

astro_path = "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/adult_grn_astrocytes_selected.h5ad"
adata = sc.read_h5ad(astro_path)

print(f"\nFirst 20 gene names:")
print(adata.var_names[:20].tolist())

print(f"\nLast 20 gene names:")
print(adata.var_names[-20:].tolist())

print(f"\nSearching for some expected human genes:")
human_genes = ['CD44', 'APOE', 'GFAP', 'GAD1', 'GAD2', 'SLC17A7', 'ACKR1']
for gene in human_genes:
    if gene in adata.var_names:
        print(f"  ✓ Found {gene}")
    else:
        print(f"  ✗ Missing {gene}")

print(f"\nChecking var (gene) metadata columns:")
print(adata.var.columns.tolist())

# Check if there's a human gene name column
if 'gene_name' in adata.var.columns or 'hgnc_symbol' in adata.var.columns:
    print("\nFound alternate gene name column!")
    
