import scanpy as sc

for name in ['astrocytes', 'gaba']:
    path = f"/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/adult_grn_{name}_selected.h5ad"
    print(f"Fixing {name}...")
    adata = sc.read_h5ad(path)
    
    # Convert to string to avoid categorical issues
    adata.var_names = adata.var['external_gene_name'].astype(str).values
    adata.var_names_make_unique()
    
    adata.write_h5ad(path)
    print(f"  ✓ Saved with human gene names")
    print(f"  Sample genes: {adata.var_names[:5].tolist()}")
