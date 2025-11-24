#!/usr/bin/env python3
"""
Preflight check: Count cells per individual, region, and cell type
to ensure data looks reasonable before running analyses
"""
import scanpy as sc
import pandas as pd
import os

base_path = "/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/"

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

print("="*80)
print("PREFLIGHT CHECK: Cell counts per individual, region, and cell type")
print("="*80)

all_data = []

for cell_type, filename in cell_types.items():
    print(f"\nLoading {cell_type}...")
    filepath = os.path.join(base_path, filename)
    adata = sc.read_h5ad(filepath, backed='r')
    
    print(f"  Total cells: {adata.n_obs:,}")
    print(f"  Individuals: {adata.obs['animal_id'].nunique()}")
    print(f"  Regions: {sorted(adata.obs['region'].unique())}")
    
    # Count cells per individual, region
    counts = adata.obs.groupby(['animal_id', 'region']).size().reset_index(name='n_cells')
    counts['cell_type'] = cell_type
    
    # Add age and sex
    meta = adata.obs.groupby('animal_id')[['age', 'sex']].first().reset_index()
    counts = counts.merge(meta, on='animal_id')
    
    all_data.append(counts)
    
    # Close file
    adata.file.close()

# Combine all counts
df = pd.concat(all_data, ignore_index=True)

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nTotal individuals: {df['animal_id'].nunique()}")
print(f"Total cell types: {df['cell_type'].nunique()}")
print(f"Total regions: {df['region'].nunique()}")

print(f"\nAge range: {df['age'].min():.1f} - {df['age'].max():.1f} years")
print(f"Sex distribution:")
print(df.groupby('sex')['animal_id'].nunique())

print(f"\nCells per cell type:")
cell_type_totals = df.groupby('cell_type')['n_cells'].sum().sort_values(ascending=False)
print(cell_type_totals)

print(f"\nCells per region:")
region_totals = df.groupby('region')['n_cells'].sum().sort_values(ascending=False)
print(region_totals)

# Check for individuals with very few cells
print("\n" + "="*80)
print("CHECKING FOR POTENTIAL ISSUES")
print("="*80)

print("\nIndividuals with < 1000 total cells:")
individual_totals = df.groupby('animal_id')['n_cells'].sum().sort_values()
low_cell_individuals = individual_totals[individual_totals < 1000]
if len(low_cell_individuals) > 0:
    for animal_id, n_cells in low_cell_individuals.items():
        age = df[df['animal_id'] == animal_id]['age'].iloc[0]
        sex = df[df['animal_id'] == animal_id]['sex'].iloc[0]
        print(f"  {animal_id}: {n_cells} cells (age={age:.1f}, sex={sex})")
else:
    print("  None - all individuals have >= 1000 cells")

print("\nCell type × region combinations with < 10 cells per individual:")
low_combos = df[df['n_cells'] < 10].copy()
if len(low_combos) > 0:
    print(f"  Found {len(low_combos)} instances")
    print(f"  Sample:")
    print(low_combos.head(20)[['animal_id', 'cell_type', 'region', 'n_cells', 'age', 'sex']])
else:
    print("  None - all combinations have >= 10 cells")

# Save full table
output_path = "/scratch/easmit31/cell_cell/u01_cell_counts_per_individual.csv"
df.to_csv(output_path, index=False)
print(f"\n✓ Saved full cell count table to:")
print(f"  {output_path}")

print("\n" + "="*80)
print("PREFLIGHT CHECK COMPLETE")
print("="*80)
print("\nReview the results above. If everything looks good, proceed with:")
print("  python /scratch/easmit31/cell_cell/run_u01_per_individual.py")
