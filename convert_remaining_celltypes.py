#!/usr/bin/env python3
"""
Convert Ensembl IDs to gene symbols using external_gene_name column
CREATES NEW FILES - does NOT overwrite originals
"""
import scanpy as sc

base_path = '/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/'

# Files to convert (NOT astrocytes/GABA - those are already done)
files_to_convert = {
    'Glutamatergic': 'adult_grn_glut_selected.h5ad',
    'MSN': 'adult_grn_MSN_selected.h5ad', 
    'OPC_Oligo': 'adult_grn_opc-olig_selected.h5ad',
    'Microglia': 'adult_grn_microglia_selected.h5ad',
    'Basket': 'adult_grn_basket_selected.h5ad',
}

for cell_type, filename in files_to_convert.items():
    print(f"\n{'='*70}")
    print(f"Converting {cell_type}...")
    
    # Load original (READ ONLY)
    adata = sc.read_h5ad(base_path + filename)
    print(f"  Original: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    
    # Use external_gene_name column (convert categorical to string)
    adata.var_names = adata.var['external_gene_name'].astype(str).values
    adata.var_names_make_unique()
    
    print(f"  Final: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    print(f"  Sample genes: {list(adata.var_names[:5])}")
    
    # Save to NEW file with _converted suffix
    output_filename = filename.replace('_selected.h5ad', '_selected_converted.h5ad')
    output_path = base_path + output_filename
    adata.write(output_path)
    print(f"  ✓ Saved NEW file: {output_filename}")

print("\n" + "="*70)
print("Done! All originals remain untouched.")
print("="*70)
