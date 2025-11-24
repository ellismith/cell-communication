#!/usr/bin/env python3
"""
Validate the logic used in run_all_pairwise.py
Checks: gene overlap, cell counts, region distribution
"""
import scanpy as sc
import pandas as pd

base_path = "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/"

# Load one example pair: Astrocyte vs GABA
print("="*70)
print("VALIDATION: Astrocyte ↔ GABA pair")
print("="*70)

# Load both
adata1 = sc.read_h5ad(base_path + "adult_grn_astrocytes_selected_converted.h5ad")
adata2 = sc.read_h5ad(base_path + "adult_grn_gaba_selected_converted.h5ad")

print("\n1. RAW DATA:")
print(f"   Astrocyte: {adata1.n_obs:,} cells × {adata1.n_vars:,} genes")
print(f"   GABA:      {adata2.n_obs:,} cells × {adata2.n_vars:,} genes")

# Check gene overlap
genes1 = set(adata1.var_names)
genes2 = set(adata2.var_names)
shared_genes = genes1 & genes2

print("\n2. GENE OVERLAP (join='inner'):")
print(f"   Astrocyte only: {len(genes1 - genes2):,} genes")
print(f"   GABA only:      {len(genes2 - genes1):,} genes")
print(f"   SHARED (used):  {len(shared_genes):,} genes")
print(f"   % retained:     {100*len(shared_genes)/len(genes1):.1f}% (Astro), {100*len(shared_genes)/len(genes2):.1f}% (GABA)")

# Tag cell types
adata1.obs["cell_type"] = "Astrocyte"
adata2.obs["cell_type"] = "GABA"

# Check regions
print("\n3. REGION DISTRIBUTION:")
print("\n   Astrocyte regions:")
print(adata1.obs['region'].value_counts().sort_index())
print(f"\n   GABA regions:")
print(adata2.obs['region'].value_counts().sort_index())

# Create combined grouping
adata1.obs["cell_type_region"] = adata1.obs["cell_type"] + "_" + adata1.obs["region"].astype(str)
adata2.obs["cell_type_region"] = adata2.obs["cell_type"] + "_" + adata2.obs["region"].astype(str)

print("\n4. COMBINED GROUPING (cell_type_region):")
print(f"\n   Astrocyte groups:")
print(adata1.obs['cell_type_region'].value_counts().sort_index())
print(f"\n   GABA groups:")
print(adata2.obs['cell_type_region'].value_counts().sort_index())

# Simulate the concat
adata_pair = sc.concat([adata1, adata2], join="inner")

print("\n5. AFTER CONCAT (join='inner'):")
print(f"   Combined: {adata_pair.n_obs:,} cells × {adata_pair.n_vars:,} genes")
print(f"   Cell type counts:")
print(adata_pair.obs['cell_type'].value_counts())
print(f"\n   Total unique groups: {adata_pair.obs['cell_type_region'].nunique()}")

print("\n6. FINAL GROUP BREAKDOWN:")
group_counts = adata_pair.obs['cell_type_region'].value_counts().sort_index()
print(group_counts)

print("\n7. SANITY CHECKS:")
print(f"   ✓ Genes retained: {adata_pair.n_vars} (should be ~15k-20k)")
print(f"   ✓ Total cells: {adata_pair.n_obs:,} (should equal sum of both types)")
print(f"   ✓ Groups created: {adata_pair.obs['cell_type_region'].nunique()} (should be ~18 for 9 regions × 2 types)")

expected_cells = adata1.n_obs + adata2.n_obs
if adata_pair.n_obs == expected_cells:
    print(f"   ✓ Cell count matches: {adata_pair.n_obs:,} = {expected_cells:,}")
else:
    print(f"   ✗ Cell count MISMATCH: {adata_pair.n_obs:,} ≠ {expected_cells:,}")

print("\n" + "="*70)
print("Validation complete!")
print("="*70)
