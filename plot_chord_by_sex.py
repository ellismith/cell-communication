#!/usr/bin/env python3
"""
plot_chord_by_sex.py
--------------------
Create chord diagrams for each sex with CONSISTENT nodes.
All 6 cell types shown in every plot.
"""
import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

base_dir = "/scratch/easmit31/cell_cell/results/pairwise_with_sex/"
out_dir = os.path.join(base_dir, "figures/chord_by_sex")
os.makedirs(out_dir, exist_ok=True)

# Load all sex-stratified results
files = glob.glob(os.path.join(base_dir, "*_results.csv"))
if not files:
    raise FileNotFoundError("No sex-stratified results found!")

print(f"Loading {len(files)} result files...")

rows = []
for f in files:
    df = pd.read_csv(f)
    if "lr_means" not in df.columns:
        continue
    
    # Extract sex, region, and cell types
    df["source_sex"] = df["source"].str.split("_").str[-1]
    df["target_sex"] = df["target"].str.split("_").str[-1]
    df["source_region"] = df["source"].str.split("_").str[-2]
    df["target_region"] = df["target"].str.split("_").str[-2]
    df["source_celltype"] = df["source"].apply(lambda x: "_".join(x.split("_")[:-2]))
    df["target_celltype"] = df["target"].apply(lambda x: "_".join(x.split("_")[:-2]))
    df["interaction"] = df["ligand_complex"] + "→" + df["receptor_complex"]
    
    rows.append(df)

data = pd.concat(rows, ignore_index=True)
print(f"Total records: {len(data):,}")

# Filter to within-sex and significant
same_sex = data[data["source_sex"] == data["target_sex"]].copy()
sig = same_sex[same_sex["magnitude_rank"] < 0.05].copy()
sig["sex_group"] = sig["source_sex"]

print(f"Significant same-sex interactions: {len(sig):,}")

# Get ALL cell types across both sexes
all_cell_types = sorted(set(sig["source_celltype"].unique()) | set(sig["target_celltype"].unique()))
print(f"\nAll cell types: {all_cell_types}")

# Get sex groups
sex_groups = sorted(sig["sex_group"].unique())
print(f"Sex groups: {sex_groups}\n")

# Create one chord diagram per sex
for sex_group in sex_groups:
    print(f"{'='*70}")
    print(f"Creating chord diagram for {sex_group}")
    print(f"{'='*70}")
    
    sex_data = sig[sig["sex_group"] == sex_group].copy()
    
    # Aggregate: mean lr_means per cell type pair
    flow = (sex_data.groupby(["source_celltype", "target_celltype"])
           .agg({
               "lr_means": "mean",
               "interaction": "nunique"
           })
           .reset_index())
    flow.columns = ["source_celltype", "target_celltype", "mean_lr", "n_interactions"]
    
    print(f"  {len(flow)} cell type flows")
    
    # Create directed graph with ALL cell types as nodes
    G = nx.DiGraph()
    
    # Add all cell types as nodes
    for cell_type in all_cell_types:
        G.add_node(cell_type)
    
    # Add edges
    for _, row in flow.iterrows():
        G.add_edge(
            row["source_celltype"],
            row["target_celltype"],
            weight=row["mean_lr"],
            n_interactions=row["n_interactions"]
        )
    
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create circular layout with FIXED positions
    pos = nx.circular_layout(sorted(G.nodes()))
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Get edge weights
    edges = list(G.edges())
    if len(edges) > 0:
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights)
        min_weight = min(weights)
        
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
    
    # Calculate node sizes
    node_sizes = []
    for node in sorted(G.nodes()):
        if G.degree(node) > 0:
            out_signal = sum([G[node][target]['weight'] for target in G.successors(node)])
            in_signal = sum([G[source][node]['weight'] for source in G.predecessors(node)])
            total = out_signal + in_signal
            if len(edges) > 0:
                node_sizes.append(1000 + 200 * (total / max_weight))
            else:
                node_sizes.append(1000)
        else:
            node_sizes.append(800)  # Smaller for unconnected nodes
    
    # Color nodes by cell type
    cell_type_colors = {
        'Astrocyte': '#FF6B6B',
        'GABA': '#4ECDC4',
        'Glutamatergic': '#45B7D1',
        'MSN': '#96CEB4',
        'OPC_Oligo': '#FFEAA7',
        'Microglia': '#DFE6E9'
    }
    node_colors = [cell_type_colors.get(node, 'lightgray') for node in sorted(G.nodes())]
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=sorted(G.nodes()),
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='black',
        linewidths=2.5,
        ax=ax
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        labels={node: node for node in sorted(G.nodes())},
        font_size=11,
        font_weight='bold',
        font_color='black',
        ax=ax
    )
    
    sex_label = "Male" if sex_group == "M" else "Female"
    ax.set_title(f"{sex_label}\nCell Type Communication Network",
                 fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add colorbar
    if len(edges) > 0 and max_weight > min_weight:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                   norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Mean LR Expression', fontsize=12)
    
    plt.tight_layout()
    
    outfile = os.path.join(out_dir, f"chord_sex_{sex_group}.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved {outfile}\n")
    plt.close()

print(f"{'='*70}")
print(f"✓ All sex chord diagrams saved to {out_dir}")
print(f"{'='*70}")
