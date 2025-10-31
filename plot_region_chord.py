#!/usr/bin/env python3
"""
plot_region_chord.py
--------------------
Create a chord diagram showing signaling flows between regions.
Width of chords = strength of signaling (average lr_means).
"""
import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.path import Path
import matplotlib.patches as mpatches

base_dir = "/scratch/easmit31/cell_cell/results/pairwise/"
out_dir = os.path.join(base_dir, "figures")
os.makedirs(out_dir, exist_ok=True)

# Load all results
files = sorted(glob.glob(os.path.join(base_dir, "*_results.csv")))
if not files:
    raise FileNotFoundError(f"No CSVs found in {base_dir}")

print(f"Loading {len(files)} result files...")
df_list = []
for f in files:
    df = pd.read_csv(f)
    if "lr_means" not in df.columns:
        continue
    df["source_region"] = df["source"].str.split("_").str[-1]
    df["target_region"] = df["target"].str.split("_").str[-1]
    df_list.append(df)

full = pd.concat(df_list, ignore_index=True)
print(f"Total records: {len(full):,}")

# Filter to significant interactions only
sig = full[full["magnitude_rank"] < 0.05].copy()
print(f"Significant interactions: {len(sig):,}")

# Aggregate: average lr_means for each region pair
flow = sig.groupby(["source_region", "target_region"])["lr_means"].agg(["mean", "sum", "count"]).reset_index()
flow.columns = ["source", "target", "mean_lr", "sum_lr", "n_interactions"]

print(f"\nRegion pairs with signaling: {len(flow)}")
print(flow.head(10))

# Get all unique regions
all_regions = sorted(set(flow["source"].unique()) | set(flow["target"].unique()))
n_regions = len(all_regions)
print(f"\nRegions: {all_regions}")

# Create matrix for chord diagram
matrix = pd.DataFrame(0.0, index=all_regions, columns=all_regions)
for _, row in flow.iterrows():
    matrix.loc[row["source"], row["target"]] = row["mean_lr"]

print(f"\nMatrix shape: {matrix.shape}")

# Normalize for visualization (use sum_lr for thickness)
flow_matrix = pd.DataFrame(0.0, index=all_regions, columns=all_regions)
for _, row in flow.iterrows():
    flow_matrix.loc[row["source"], row["target"]] = row["sum_lr"]

# Simple chord diagram using network-style plot
import networkx as nx

# Create directed graph
G = nx.DiGraph()

# Add edges with weights
for _, row in flow.iterrows():
    if row["mean_lr"] > 0:
        G.add_edge(
            row["source"], 
            row["target"], 
            weight=row["mean_lr"],
            n_interactions=row["n_interactions"]
        )

print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Create circular layout
pos = nx.circular_layout(G)

# Plot
fig, ax = plt.subplots(figsize=(14, 14))

# Draw edges with width proportional to signaling strength
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
max_weight = max(weights)

# Normalize weights for visualization
edge_widths = [5 * (w / max_weight) for w in weights]

nx.draw_networkx_edges(
    G, pos,
    width=edge_widths,
    alpha=0.6,
    edge_color=weights,
    edge_cmap=plt.cm.viridis,
    arrows=True,
    arrowsize=20,
    arrowstyle='->',
    connectionstyle='arc3,rad=0.1',
    ax=ax
)

# Draw nodes
node_sizes = []
for node in G.nodes():
    # Size based on total outgoing + incoming signal
    out_signal = sum([G[node][target]['weight'] for target in G.successors(node)])
    in_signal = sum([G[source][node]['weight'] for source in G.predecessors(node)])
    node_sizes.append(500 + 100 * (out_signal + in_signal))

nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color='lightblue',
    edgecolors='black',
    linewidths=2,
    ax=ax
)

# Draw labels
nx.draw_networkx_labels(
    G, pos,
    font_size=12,
    font_weight='bold',
    ax=ax
)

ax.set_title("Cell-Cell Communication Network Across Brain Regions\n(Edge width = signaling strength)", 
             fontsize=16, fontweight='bold', pad=20)
ax.axis('off')

# Add colorbar for edge colors
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Mean LR Expression', fontsize=12)

plt.tight_layout()
outfile = os.path.join(out_dir, "region_communication_network.png")
plt.savefig(outfile, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved network diagram to {outfile}")

# Also create a summary table
summary = flow.groupby("source").agg({
    "mean_lr": "mean",
    "n_interactions": "sum"
}).round(3)
summary.columns = ["avg_signaling_strength", "total_interactions"]
summary = summary.sort_values("avg_signaling_strength", ascending=False)

print("\nRegion signaling summary:")
print(summary)

summary.to_csv(os.path.join(out_dir, "region_signaling_summary.csv"))
print(f"✓ Saved summary to region_signaling_summary.csv")

# Top signaling pairs
top_pairs = flow.nlargest(20, "mean_lr")[["source", "target", "mean_lr", "n_interactions"]]
print("\nTop 20 region pairs by signaling strength:")
print(top_pairs)
top_pairs.to_csv(os.path.join(out_dir, "top_region_pairs.csv"), index=False)
print(f"✓ Saved top pairs to top_region_pairs.csv")
