#!/usr/bin/env python3
"""
Analyze L-R pairs that actually exist in the data
Keep them in the order from the image
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the found L-R pairs (in image order)
found_pairs_df = pd.read_csv('lr_pairs_found_in_order.csv')
lr_pairs_ordered = found_pairs_df['lr_pair'].tolist()

print(f"Using {len(lr_pairs_ordered)} L-R pairs found in data (in image order)")

# Define cell type order
cell_type_order = [
    'OPC_Oligo', 'Microglia', 'Astrocyte', 'Ependymal', 'Vascular',
    'GABA', 'Glutamatergic', 'MSN', 'Basket', 'Cerebellar', 'Midbrain'
]

# Generate all pairwise combinations
celltype_pairs_ordered = []
for source in cell_type_order:
    for target in cell_type_order:
        celltype_pairs_ordered.append(f"{source}_{target}")

print(f"Using {len(celltype_pairs_ordered)} cell type pair combinations (11×11=121)")

# Load all individual results
results_dir = "results/u01_per_individual"
result_files = [f for f in os.listdir(results_dir) if f.endswith('_results.csv')]

print(f"\nLoading {len(result_files)} individuals...")

all_data = []
for idx, result_file in enumerate(result_files):
    if idx % 10 == 0:
        print(f"  Loading {idx}/{len(result_files)}...")
    df = pd.read_csv(f"{results_dir}/{result_file}")
    all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)
print(f"Loaded {len(combined):,} total interactions")

# Extract cell types
combined['source_celltype'] = combined['source'].str.rsplit('_', n=1).str[0]
combined['target_celltype'] = combined['target'].str.rsplit('_', n=1).str[0]
combined['celltype_pair'] = combined['source_celltype'] + '_' + combined['target_celltype']
combined['lr_pair'] = combined['ligand_complex'] + '|' + combined['receptor_complex']

# Filter to specified pairs
combined_filtered = combined[
    (combined['celltype_pair'].isin(celltype_pairs_ordered)) &
    (combined['lr_pair'].isin(lr_pairs_ordered))
].copy()

print(f"\nFiltered to {len(combined_filtered):,} interactions")

# Calculate mean lr_means
pivot_data = combined_filtered.groupby(['celltype_pair', 'lr_pair'])['lr_means'].mean().reset_index()

# Create matrix
heatmap_matrix = pivot_data.pivot(index='celltype_pair', columns='lr_pair', values='lr_means')

# Reorder to match our specified orders
celltype_pairs_present = [cp for cp in celltype_pairs_ordered if cp in heatmap_matrix.index]
lr_pairs_present = [lp for lp in lr_pairs_ordered if lp in heatmap_matrix.columns]

heatmap_matrix = heatmap_matrix.reindex(index=celltype_pairs_present, columns=lr_pairs_present)

print(f"\nHeatmap dimensions:")
print(f"  Rows (cell type pairs): {len(celltype_pairs_present)}")
print(f"  Columns (L-R pairs): {len(lr_pairs_present)}")

# Plot
fig, ax = plt.subplots(figsize=(max(20, len(lr_pairs_present) * 0.8), 
                                 max(18, len(celltype_pairs_present) * 0.25)))

sns.heatmap(heatmap_matrix, cmap='RdYlBu_r', center=heatmap_matrix.mean().mean(), 
            cbar_kws={'label': 'Mean lr_means'}, 
            linewidths=0.5, linecolor='lightgray',
            ax=ax, annot=False)

ax.set_xlabel('Ligand|Receptor Pair (Image Order)', fontsize=12, fontweight='bold')
ax.set_ylabel('Cell Type Pair: Source → Target', fontsize=12, fontweight='bold')
ax.set_title('Communication Patterns: Specific L-R Pairs (Found in Data)', 
             fontsize=14, fontweight='bold', pad=20)

plt.xticks(rotation=90, ha='center', fontsize=7)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()

output_path = 'results/specific_lr_celltype_heatmap.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved heatmap to {output_path}")

# Save data
heatmap_matrix.to_csv('results/specific_lr_celltype_matrix.csv')
print(f"✓ Saved matrix to results/specific_lr_celltype_matrix.csv")
