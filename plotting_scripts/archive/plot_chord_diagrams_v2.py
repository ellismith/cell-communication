#!/usr/bin/env python3
"""
Create chord diagrams showing cell-cell communication strength within each region.
Arrow thickness = mean lr_means for all interactions between those cell types.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.lines as mlines

parser = argparse.ArgumentParser()
parser.add_argument('--region', required=True, help='Region to analyze (e.g., HIP, ACC)')
parser.add_argument('--min_lr_means', type=float, default=1.0, help='Min lr_means threshold')
parser.add_argument('--top_n', type=int, default=None, help='Show only top N interactions')
args = parser.parse_args()

region = args.region
min_lr_means = args.min_lr_means
top_n = args.top_n

print("="*80)
print(f"CREATING CHORD DIAGRAM FOR {region}")
print("="*80)

# Find files for this region
region_files = glob.glob(f'/scratch/easmit31/cell_cell/results/per_animal_louvain_corrected/*_{region}_louvain_results.csv')
print(f"Found {len(region_files)} files for {region}")

# Collect data
celltype_pair_sums = defaultdict(float)
celltype_pair_counts = defaultdict(int)

print("Processing files...")
for i, f in enumerate(region_files):
    if i % 10 == 0:
        print(f"  File {i+1}/{len(region_files)}...")
    
    chunk_iter = pd.read_csv(
        f,
        usecols=['source', 'target', 'lr_means'],
        chunksize=1000000
    )
    
    for chunk in chunk_iter:
        # Filter by threshold
        chunk = chunk[chunk['lr_means'] >= min_lr_means]
        
        if chunk.empty:
            continue
        
        # Extract base cell types (remove Louvain cluster numbers)
        chunk['source_celltype'] = chunk['source'].str.rsplit('_', n=1).str[0]
        chunk['target_celltype'] = chunk['target'].str.rsplit('_', n=1).str[0]
        
        # Aggregate by cell type pair
        pair_agg = chunk.groupby(['source_celltype', 'target_celltype'])['lr_means'].agg(['sum', 'count'])
        
        for (src, tgt), row in pair_agg.iterrows():
            pair_key = f"{src}→{tgt}"
            celltype_pair_sums[pair_key] += row['sum']
            celltype_pair_counts[pair_key] += row['count']

# Calculate means
celltype_interactions = []
for pair_key, total in celltype_pair_sums.items():
    src, tgt = pair_key.split('→')
    mean_val = total / celltype_pair_counts[pair_key]
    celltype_interactions.append({
        'source': src,
        'target': tgt,
        'mean_lr_means': mean_val,
        'count': celltype_pair_counts[pair_key]
    })

interactions_df = pd.DataFrame(celltype_interactions).sort_values('mean_lr_means', ascending=False)

print(f"\nTotal cell type interactions: {len(interactions_df)}")

# Filter to top N if requested
if top_n:
    interactions_df = interactions_df.head(top_n)
    print(f"Showing top {top_n} interactions")

print("\nTop 20 interactions:")
print(interactions_df.head(20).to_string(index=False))

# Get unique cell types
all_celltypes = sorted(set(interactions_df['source']) | set(interactions_df['target']))
n_celltypes = len(all_celltypes)

print(f"\nCell types involved: {n_celltypes}")
print(f"  {', '.join(all_celltypes)}")

# Create circular chord diagram
fig, ax = plt.subplots(figsize=(14, 14))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')

# Position cell types in a circle
radius = 1.0
angles = np.linspace(0, 2*np.pi, n_celltypes, endpoint=False)
celltype_positions = {}
colors = plt.cm.Set3(np.linspace(0, 1, n_celltypes))
celltype_colors = {}

for i, (celltype, angle) in enumerate(zip(all_celltypes, angles)):
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    celltype_positions[celltype] = (x, y, angle)
    celltype_colors[celltype] = colors[i]
    
    # Draw cell type label
    label_x = 1.15 * x
    label_y = 1.15 * y
    ha = 'right' if x < 0 else 'left'
    va = 'center'
    ax.text(label_x, label_y, celltype, ha=ha, va=va, fontsize=10, fontweight='bold')
    
    # Draw small circle for cell type
    circle = plt.Circle((x, y), 0.08, color=celltype_colors[celltype], alpha=0.8, zorder=3)
    ax.add_patch(circle)

# Normalize arrow widths
max_lr_means = interactions_df['mean_lr_means'].max()
min_lr_means_val = interactions_df['mean_lr_means'].min()

# Draw connections (curved arrows)
for _, row in interactions_df.iterrows():
    src = row['source']
    tgt = row['target']
    strength = row['mean_lr_means']
    
    if src not in celltype_positions or tgt not in celltype_positions:
        continue
    
    x1, y1, _ = celltype_positions[src]
    x2, y2, _ = celltype_positions[tgt]
    
    # Normalize line width (0.5 to 5)
    width = 0.5 + 4.5 * (strength - min_lr_means_val) / (max_lr_means - min_lr_means_val)
    
    # Create curved connection (Bezier curve through center)
    # Control point at center with slight offset
    mid_x = (x1 + x2) / 2 * 0.3
    mid_y = (y1 + y2) / 2 * 0.3
    
    # Draw curved arrow
    from matplotlib.patches import FancyArrowPatch
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad=.3",
        arrowstyle='->,head_width=0.3,head_length=0.3',
        color=celltype_colors[src],
        alpha=0.4,
        linewidth=width,
        zorder=1
    )
    ax.add_patch(arrow)

# Title
ax.set_title(f'Cell-Cell Communication in {region}\n(Arrow thickness = mean lr_means, threshold ≥ {min_lr_means})', 
             fontsize=16, fontweight='bold', pad=20)

# Save
output_dir = Path('/scratch/easmit31/cell_cell/results/louvain_lm_analysis/chord_diagrams')
output_dir.mkdir(exist_ok=True, parents=True)

output_file = output_dir / f'{region.lower()}_chord_diagram_threshold{min_lr_means:.1f}.png'
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_file}")

# Also save the data
data_file = output_dir / f'{region.lower()}_celltype_interactions_threshold{min_lr_means:.1f}.csv'
interactions_df.to_csv(data_file, index=False)
print(f"✓ Saved data: {data_file}")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
