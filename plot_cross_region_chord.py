#!/usr/bin/env python3
"""
plot_cross_region_chord.py
--------------------------
Create a chord diagram showing cross-region signaling flows.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

base_dir = "/scratch/easmit31/cell_cell/results/pairwise/cross_region_analysis/"
out_dir = os.path.join(base_dir, "figures")
os.makedirs(out_dir, exist_ok=True)

# Load cross-region summary
summary_file = os.path.join(base_dir, "cross_region_summary.csv")
if not os.path.exists(summary_file):
    raise FileNotFoundError(f"Run analyze_across_regions.py first!")

flow = pd.read_csv(summary_file)
print(f"Loaded {len(flow)} region-to-region flows")
print(flow.head(10))

# Create directed graph
G = nx.DiGraph()

# Add edges with weights
for _, row in flow.iterrows():
    G.add_edge(
        row["source_region"],
        row["target_region"],
        weight=row["mean_lr"],
        n_interactions=row["n_unique_interactions"]
    )

print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Get all regions
all_regions = sorted(G.nodes())
print(f"Regions: {all_regions}")

# Create circular layout
pos = nx.circular_layout(G)

# Plot
fig, ax = plt.subplots(figsize=(16, 16))

# Get edge weights
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
max_weight = max(weights)
min_weight = min(weights)

# Normalize weights for visualization
edge_widths = [2 + 10 * ((w - min_weight) / (max_weight - min_weight)) for w in weights]

# Draw edges with curved arrows
nx.draw_networkx_edges(
    G, pos,
    width=edge_widths,
    alpha=0.6,
    edge_color=weights,
    edge_cmap=plt.cm.plasma,
    arrows=True,
    arrowsize=25,
    arrowstyle='->',
    connectionstyle='arc3,rad=0.15',
    ax=ax,
    node_size=3000
)

# Calculate node sizes based on total signaling (outgoing + incoming)
node_sizes = []
for node in G.nodes():
    out_signal = sum([G[node][target]['weight'] for target in G.successors(node)])
    in_signal = sum([G[source][node]['weight'] for source in G.predecessors(node)])
    node_sizes.append(1000 + 200 * (out_signal + in_signal) / max_weight)

# Draw nodes
nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color='lightcoral',
    edgecolors='darkred',
    linewidths=3,
    ax=ax
)

# Draw labels
nx.draw_networkx_labels(
    G, pos,
    font_size=14,
    font_weight='bold',
    font_color='white',
    ax=ax
)

ax.set_title("Cross-Region Cell-Cell Communication Network\n(Edge width = signaling strength)",
             fontsize=18, fontweight='bold', pad=20)
ax.axis('off')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                           norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Mean LR Expression', fontsize=12)

plt.tight_layout()

outfile = os.path.join(out_dir, "cross_region_chord_network.png")
plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ Saved chord diagram to {outfile}")

# Print top flows
print("\nTop 10 strongest cross-region flows:")
top_flows = flow.nlargest(10, "mean_lr")
for _, row in top_flows.iterrows():
    print(f"  {row['source_region']} → {row['target_region']}: {row['mean_lr']:.2f} ({row['n_unique_interactions']} interactions)")

