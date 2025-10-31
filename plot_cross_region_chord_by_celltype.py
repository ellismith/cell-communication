#!/usr/bin/env python3
"""
plot_cross_region_chord_by_celltype.py
--------------------------------------
Create a chord diagram for EACH cell type pair showing their
cross-region signaling patterns.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

base_dir = "/scratch/easmit31/cell_cell/results/pairwise/cross_region_analysis/"
out_dir = os.path.join(base_dir, "figures/chord_by_celltype")
os.makedirs(out_dir, exist_ok=True)

# Load detailed cross-region data
data_file = os.path.join(base_dir, "cross_region_effects_detailed.csv")
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Run analyze_across_regions.py first!")

data = pd.read_csv(data_file)
print(f"Loaded {len(data):,} cross-region interactions")

# Filter to significant
sig_data = data[data["magnitude_rank"] < 0.05].copy()
print(f"Significant: {len(sig_data):,}")

# Get all unique cell type pairs
cell_pairs = sorted(sig_data["pair_name"].unique())
print(f"\nFound {len(cell_pairs)} cell type pairs\n")

# Create one chord diagram per cell type pair
for pair_name in cell_pairs:
    print(f"{'='*70}")
    print(f"Creating chord diagram for {pair_name}")
    print(f"{'='*70}")
    
    pair_data = sig_data[sig_data["pair_name"] == pair_name].copy()
    
    if len(pair_data) == 0:
        print(f"⚠️ No data, skipping\n")
        continue
    
    # Aggregate: mean lr_means per source_region → target_region
    flow = (pair_data.groupby(["source_region", "target_region"])
           .agg({
               "lr_means": "mean",
               "interaction": "nunique"
           })
           .reset_index())
    flow.columns = ["source_region", "target_region", "mean_lr", "n_interactions"]
    
    print(f"  {len(flow)} region-to-region flows")
    
    if len(flow) < 2:
        print(f"⚠️ Not enough flows for visualization, skipping\n")
        continue
    
    # Create directed graph
    G = nx.DiGraph()
    
    for _, row in flow.iterrows():
        G.add_edge(
            row["source_region"],
            row["target_region"],
            weight=row["mean_lr"],
            n_interactions=row["n_interactions"]
        )
    
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create circular layout
    pos = nx.circular_layout(G)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Get edge weights
    edges = list(G.edges())
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights)
    min_weight = min(weights)
    
    # Normalize weights
    edge_widths = [2 + 8 * ((w - min_weight) / (max_weight - min_weight)) for w in weights]
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.6,
        edge_color=weights,
        edge_cmap=plt.cm.viridis,
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.15',
        ax=ax,
        node_size=2500
    )
    
    # Calculate node sizes
    node_sizes = []
    for node in G.nodes():
        out_signal = sum([G[node][target]['weight'] for target in G.successors(node)])
        in_signal = sum([G[source][node]['weight'] for source in G.predecessors(node)])
        total = out_signal + in_signal
        node_sizes.append(800 + 150 * (total / max_weight) if max_weight > 0 else 800)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color='skyblue',
        edgecolors='navy',
        linewidths=2.5,
        ax=ax
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=12,
        font_weight='bold',
        font_color='black',
        ax=ax
    )
    
    ax.set_title(f"{pair_name.replace('_', ' ↔ ')}\nCross-Region Communication Network",
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Mean LR Expression', fontsize=11)
    
    plt.tight_layout()
    
    outfile = os.path.join(out_dir, f"chord_{pair_name}.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved {outfile}\n")
    plt.close()

print(f"{'='*70}")
print(f"✓ All chord diagrams saved to {out_dir}")
print(f"{'='*70}")
