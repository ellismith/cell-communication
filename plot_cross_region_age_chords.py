#!/usr/bin/env python3
"""
plot_cross_region_age_chords.py
--------------------------------
Create chord diagrams showing cross-region communication for each age group.
Nodes = regions, edges = inter-regional signaling.
"""
import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

base_dir = "/scratch/easmit31/cell_cell/results/pairwise_with_age/"
out_dir = os.path.join(base_dir, "figures/cross_region_age_chords")
os.makedirs(out_dir, exist_ok=True)

# Load all age-stratified results
files = glob.glob(os.path.join(base_dir, "*_results.csv"))
if not files:
    raise FileNotFoundError("No age-stratified results found!")

print(f"Loading {len(files)} result files...")

rows = []
for f in files:
    df = pd.read_csv(f)
    if "lr_means" not in df.columns:
        continue
    
    df["source_age"] = df["source"].str.split("_").str[-1]
    df["target_age"] = df["target"].str.split("_").str[-1]
    df["source_region"] = df["source"].str.split("_").str[-2]
    df["target_region"] = df["target"].str.split("_").str[-2]
    df["interaction"] = df["ligand_complex"] + "→" + df["receptor_complex"]
    
    rows.append(df)

data = pd.concat(rows, ignore_index=True)
print(f"Total records: {len(data):,}")

# Filter to cross-region, same age, significant
cross = data[(data["source_age"] == data["target_age"]) & 
             (data["source_region"] != data["target_region"])].copy()
sig = cross[cross["magnitude_rank"] < 0.05].copy()
sig["age_group"] = sig["source_age"]

print(f"Significant cross-region: {len(sig):,}")

# Define age order
age_order = ['young', 'medium', 'old']
age_groups = [age for age in age_order if age in sig["age_group"].unique()]
print(f"\nAge groups (ordered): {age_groups}\n")

# Get all regions for consistent layout
all_regions = sorted(sig["source_region"].unique())

# Create chord for each age group
for age in age_groups:
    print(f"{'='*70}")
    print(f"Creating cross-region chord for {age}")
    print(f"{'='*70}")
    
    age_data = sig[sig["age_group"] == age].copy()
    
    # Aggregate: mean lr_means per region pair
    flow = (age_data.groupby(["source_region", "target_region"])["lr_means"]
           .mean()
           .reset_index())
    
    print(f"  {len(flow)} region-to-region flows")
    
    # Create directed graph with all regions
    G = nx.DiGraph()
    for region in all_regions:
        G.add_node(region)
    
    for _, row in flow.iterrows():
        G.add_edge(
            row["source_region"],
            row["target_region"],
            weight=row["lr_means"]
        )
    
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Fixed circular layout
    pos = nx.circular_layout(sorted(G.nodes()))
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Edges
    edges = list(G.edges())
    if len(edges) > 0:
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights)
        min_weight = min(weights)
        
        if max_weight > min_weight:
            edge_widths = [2 + 10 * ((w - min_weight) / (max_weight - min_weight)) for w in weights]
        else:
            edge_widths = [6] * len(weights)
        
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
    
    # Node sizes
    node_sizes = []
    for node in sorted(G.nodes()):
        if G.degree(node) > 0 and len(edges) > 0:
            out_signal = sum([G[node][target]['weight'] for target in G.successors(node)])
            in_signal = sum([G[source][node]['weight'] for source in G.predecessors(node)])
            total = out_signal + in_signal
            node_sizes.append(1000 + 200 * (total / max_weight))
        else:
            node_sizes.append(800)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=sorted(G.nodes()),
        node_size=node_sizes,
        node_color='lightcoral',
        edgecolors='darkred',
        linewidths=3,
        ax=ax
    )
    
    # Labels
    nx.draw_networkx_labels(
        G, pos,
        labels={node: node for node in sorted(G.nodes())},
        font_size=12,
        font_weight='bold',
        font_color='white',
        ax=ax
    )
    
    ax.set_title(f"{age.capitalize()} Age Group\nCross-Region Communication Network",
                 fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Colorbar
    if len(edges) > 0 and max_weight > min_weight:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                   norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Mean LR Expression', fontsize=12)
    
    plt.tight_layout()
    
    outfile = os.path.join(out_dir, f"cross_region_chord_{age}.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved {outfile}\n")
    plt.close()

print(f"{'='*70}")
print(f"✓ All cross-region age chords saved to {out_dir}")
print(f"{'='*70}")
