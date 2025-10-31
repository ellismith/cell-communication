#!/usr/bin/env python3
"""
CellChat Analysis: Astrocyte-GABA Communication
Using liana-py for cell-cell communication analysis

This script:
1. Loads adult astrocyte and GABA data
2. Combines them into a single dataset
3. Runs cell-cell communication analysis
4. Identifies astrocyte<->GABA signaling patterns
"""

import scanpy as sc
import pandas as pd
import numpy as np
import liana as li
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("CellChat Analysis: Astrocyte-GABA Communication")
print("="*70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[Step 1] Loading data...")

astro_path = "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/adult_grn_astrocytes_selected.h5ad"
gaba_path = "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/adult_grn_gaba_selected.h5ad"

adata_astro = sc.read_h5ad(astro_path)
adata_gaba = sc.read_h5ad(gaba_path)

print(f"  Astrocytes: {adata_astro.n_obs:,} cells")
print(f"  GABA neurons: {adata_gaba.n_obs:,} cells")

# ============================================================================
# 2. ADD CELL TYPE LABELS
# ============================================================================
print("\n[Step 2] Adding cell type labels...")

adata_astro.obs['cell_type'] = 'Astrocyte'
adata_gaba.obs['cell_type'] = 'GABA_neuron'

# ============================================================================
# 3. COMBINE DATASETS
# ============================================================================
print("\n[Step 3] Combining datasets...")

# Concatenate (keep only shared genes)
adata_combined = sc.concat(
    [adata_astro, adata_gaba],
    join='inner',  # Only keep genes present in both
    label='source',
    keys=['astro', 'gaba']
)

print(f"  Combined: {adata_combined.n_obs:,} cells x {adata_combined.n_vars:,} genes")
print(f"  Cell type distribution:")
print(adata_combined.obs['cell_type'].value_counts())

# ============================================================================
# 4. PREPARE FOR LIANA
# ============================================================================
print("\n[Step 4] Preparing data for cell-cell communication analysis...")

# Check if data needs normalization
max_val = adata_combined.X.max() if hasattr(adata_combined.X, 'max') else adata_combined.X.data.max()
print(f"  Max expression value: {max_val:.2f}")

if max_val > 100:
    print("  ⚠️  Data appears to be raw counts - normalizing...")
    sc.pp.normalize_total(adata_combined, target_sum=1e4)
    sc.pp.log1p(adata_combined)
else:
    print("  ✓ Data appears normalized")

# ============================================================================
# 5. RUN LIANA (CELL-CELL COMMUNICATION)
# ============================================================================
print("\n[Step 5] Running cell-cell communication analysis...")
print("  Using CellChat database (human ligand-receptor pairs)")

# Run liana with CellChat method
# This identifies ligand-receptor interactions between cell types
li.mt.rank_aggregate(
    adata_combined,
    groupby='cell_type',
    resource_name='cellchatdb',  # Use CellChat's human L-R database
    expr_prop=0.1,  # Ligand/receptor expressed in >10% of cells
    use_raw=False,
    n_perms=100,  # Number of permutations for significance testing
    verbose=True
)

print("  ✓ Communication analysis complete!")

# Results are stored in adata_combined.uns['liana_res']
comm_results = adata_combined.uns['liana_res']

print(f"\n  Total interactions tested: {len(comm_results):,}")
print(f"  Significant interactions (magnitude_rank < 0.05): {(comm_results['magnitude_rank'] < 0.05).sum()}")

# ============================================================================
# 6. FILTER AND EXAMINE RESULTS
# ============================================================================
print("\n[Step 6] Examining communication patterns...")

# Filter for significant interactions
significant = comm_results[comm_results['magnitude_rank'] < 0.05].copy()
significant = significant.sort_values('magnitude_rank')

print(f"\n  Significant interactions: {len(significant)}")

# Separate by direction
astro_to_gaba = significant[
    (significant['source'] == 'Astrocyte') & 
    (significant['target'] == 'GABA_neuron')
]

gaba_to_astro = significant[
    (significant['source'] == 'GABA_neuron') & 
    (significant['target'] == 'Astrocyte')
]

print(f"\n  Astrocyte → GABA: {len(astro_to_gaba)} interactions")
print(f"  GABA → Astrocyte: {len(gaba_to_astro)} interactions")

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print("\n[Step 7] Saving results...")

output_dir = "/scratch/easmit31/cell_cell/results"
import os
os.makedirs(output_dir, exist_ok=True)

# Save all significant interactions
significant.to_csv(f"{output_dir}/cellchat_significant_interactions.csv", index=False)

# Save direction-specific results
astro_to_gaba.to_csv(f"{output_dir}/cellchat_astro_to_gaba.csv", index=False)
gaba_to_astro.to_csv(f"{output_dir}/cellchat_gaba_to_astro.csv", index=False)

print(f"  ✓ Results saved to {output_dir}/")

# ============================================================================
# 8. TOP INTERACTIONS SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TOP ASTROCYTE → GABA INTERACTIONS")
print("="*70)

if len(astro_to_gaba) > 0:
    top_astro_gaba = astro_to_gaba.head(10)
    for idx, row in top_astro_gaba.iterrows():
        print(f"{row['ligand']:10s} → {row['receptor']:10s}  (rank: {row['magnitude_rank']:.4f})")
else:
    print("No significant interactions found")

print("\n" + "="*70)
print("TOP GABA → ASTROCYTE INTERACTIONS")
print("="*70)

if len(gaba_to_astro) > 0:
    top_gaba_astro = gaba_to_astro.head(10)
    for idx, row in top_gaba_astro.iterrows():
        print(f"{row['ligand']:10s} → {row['receptor']:10s}  (rank: {row['magnitude_rank']:.4f})")
else:
    print("No significant interactions found")

# ============================================================================
# 9. CREATE BASIC VISUALIZATIONS
# ============================================================================
print("\n[Step 8] Creating visualizations...")

fig_dir = "/scratch/easmit31/cell_cell/figures"
os.makedirs(fig_dir, exist_ok=True)

# Dotplot of top interactions
if len(significant) > 0:
    li.pl.dotplot(
        adata_combined,
        colour='magnitude_rank',
        size='cellchat_score',
        source_labels=['Astrocyte', 'GABA_neuron'],
        target_labels=['Astrocyte', 'GABA_neuron'],
        top_n=20,  # Show top 20 interactions
        orderby='magnitude_rank',
        orderby_ascending=True,
        figure_size=(8, 10)
    )
    plt.savefig(f"{fig_dir}/cellchat_dotplot_top20.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Dotplot saved: {fig_dir}/cellchat_dotplot_top20.png")
    plt.close()

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"\nResults saved to: {output_dir}/")
print(f"Figures saved to: {fig_dir}/")
print("\nNext steps:")
print("  1. Review significant interactions in CSV files")
print("  2. Run age/sex stratified analyses")
print("  3. Explore specific pathways of interest")
