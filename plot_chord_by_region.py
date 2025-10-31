#!/usr/bin/env python3
"""
plot_chord_by_region.py
-----------------------
Create a chord diagram for EACH region showing cell type communication
within that region.
"""
import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

base_dir = "/scratch/easmit31/cell_cell/results/pairwise/"
out_dir = os.path.join(base_dir, "figures/chord_by_region")
os.makedirs(out_dir, exist_ok=True)

# Load all pairwise results
files = glob.glob(os.path.join(base_dir, "*_results.csv"))
if not files:
    raise FileNotFoundError("No results files found!")

print(f"Loading {len(files)} result files...")

rows = []
for f in files:
    df = pd.read_csv(f)
    if "lr_means" not in df.columns:
        continue
    
    # Extract regions and cell types
    df["source_region"] = df["source"].str.split("_").str[-1]
    df["target_region"] = df["target"].str.split("_").str[-1]
    df["source_celltype"] = df["source"].apply(lambda x: "_".join(x.split("_")[:-1]))
    df["target_celltype"] = df["target"].apply(lambda x: "_".join(x.split("_")[:-1]))
    
    # Create interaction column
    df["interaction"] = df["ligand_complex"] + "→" + df["receptor_complex"]
    
    rows.append(df)

data = pd.concat(rows, ignore_index=True)
print(f"Total records: {len(data):,}")

# Filter to within-region and significant
within = data[data["source_region"] == data["target_region"]].copy()
sig = within[within["magnitude_rank"] < 0.05].copy()
sig["region"] = sig["source_region"]

print(f"Significant within-region: {len(sig):,}")

# Get all unique regions
regions = sorted(sig["region"].unique())
print(f"\nRegions: {regions}\n")

# Create one chord diagram per region
for region in regions:
    print(f"{'='*70}")
    print(f"Creating chord diagram for {region}")
    print(f"{'='*70}")
    
    region_data = sig[sig["region"] == region].copy()
    
    # Aggregate: mean lr_means per cell type pair
    flow = (region_data.groupby(["source_celltype", "target_celltype"])
           .agg({
               "lr_means": "mean",
               "interaction": "nunique"
           })
           .reset_index())
    flow.columns = ["source_celltype", "target_celltype", "mean_lr", "n_interactions"]
    
    print(f"  {len(flow)} cell type flows")
    
    if len(flow) < 2:
        print(f"⚠️ Not enough flows, skipping\n")
        continue
    
    # Create directed graph
    G = nx.DiGraph()
    
    for _, row in flow.iterrows():
        G.add_edge(
            row["source_celltype"],
            row["target_celltype"],
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
    min_weight = min(weights) if len(weights) > 0 else 0
    
    # Normalize weights
    if max_weight > min_weight:
        edge_widths = [2 + 10 * ((w - min_weight) / (max_weight - min_weight)) for w in weights]
    else:
        edge_widths = [6] * len(weights)
    
    # Draw edges
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
    
    # Calculate node sizes based on signaling activity
    node_sizes = []
    for node in G.nodes():
        out_signal = sum([G[node][target]['weight'] for target in G.successors(node)])
        in_signal = sum([G[source][node]['weight'] for source in G.predecessors(node)])
        total = out_signal + in_signal
        node_sizes.append(1000 + 200 * (total / max_weight) if max_weight > 0 else 1000)
    
    # Color nodes by cell type
    cell_type_colors = {
        'Astrocyte': '#FF6B6B',
        'GABA': '#4ECDC4',
        'Glutamatergic': '#45B7D1',
        'MSN': '#96CEB4',
        'OPC_Oligo': '#FFEAA7',
        'Microglia': '#DFE6E9'
    }
    node_colors = [cell_type_colors.get(node, 'lightgray') for node in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='black',
        linewidths=2.5,
        ax=ax
    )
    
    # Draw labels with BLACK text
    nx.draw_networkx_labels(
        G, pos,
        font_size=11,
        font_weight='bold',
        font_color='black',  # Changed from white to black
        ax=ax
    )
    
    ax.set_title(f"{region} Region\nCell Type Communication Network",
                 fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add colorbar
    if max_weight > min_weight:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                   norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Mean LR Expression', fontsize=12)
    
    plt.tight_layout()
    
    outfile = os.path.join(out_dir, f"chord_{region}.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved {outfile}\n")
    plt.close()

print(f"{'='*70}")
print(f"✓ All regional chord diagrams saved to {out_dir}")
print(f"{'='*70}")
