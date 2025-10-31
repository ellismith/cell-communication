#!/usr/bin/env python3
"""
plot_chord_by_age_and_region.py
--------------------------------
Create chord diagrams for each age group within each region.
Shows how cell type communication varies by age within specific brain regions.
"""
import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

base_dir = "/scratch/easmit31/cell_cell/results/pairwise_with_age/"
out_dir = os.path.join(base_dir, "figures/chord_by_age_and_region")
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
    
    # Extract age, region, and cell types
    df["source_age"] = df["source"].str.split("_").str[-1]
    df["target_age"] = df["target"].str.split("_").str[-1]
    df["source_region"] = df["source"].str.split("_").str[-2]
    df["target_region"] = df["target"].str.split("_").str[-2]
    df["source_celltype"] = df["source"].apply(lambda x: "_".join(x.split("_")[:-2]))
    df["target_celltype"] = df["target"].apply(lambda x: "_".join(x.split("_")[:-2]))
    df["interaction"] = df["ligand_complex"] + "→" + df["receptor_complex"]
    
    rows.append(df)

data = pd.concat(rows, ignore_index=True)
print(f"Total records: {len(data):,}")

# Filter to within-region, within-age-group, and significant
within = data[(data["source_age"] == data["target_age"]) & 
              (data["source_region"] == data["target_region"])].copy()
sig = within[within["magnitude_rank"] < 0.05].copy()
sig["age_group"] = sig["source_age"]
sig["region"] = sig["source_region"]

print(f"Significant within-region, within-age interactions: {len(sig):,}")

# Get ALL cell types for consistent nodes
all_cell_types = sorted(set(sig["source_celltype"].unique()) | set(sig["target_celltype"].unique()))
print(f"\nAll cell types: {all_cell_types}")

# Get age groups and regions
age_groups = sorted(sig["age_group"].unique())
regions = sorted(sig["region"].unique())
print(f"Age groups: {age_groups}")
print(f"Regions: {regions}\n")

# Create chord diagram for each age × region combination
for age_group in age_groups:
    for region in regions:
        print(f"{'='*70}")
        print(f"Creating chord: {region} - {age_group}")
        print(f"{'='*70}")
        
        subset = sig[(sig["age_group"] == age_group) & (sig["region"] == region)].copy()
        
        if len(subset) == 0:
            print(f"⚠️ No data, skipping\n")
            continue
        
        # Aggregate
        flow = (subset.groupby(["source_celltype", "target_celltype"])
               .agg({
                   "lr_means": "mean",
                   "interaction": "nunique"
               })
               .reset_index())
        flow.columns = ["source_celltype", "target_celltype", "mean_lr", "n_interactions"]
        
        print(f"  {len(flow)} cell type flows")
        
        # Create graph with all cell types
        G = nx.DiGraph()
        for cell_type in all_cell_types:
            G.add_node(cell_type)
        
        for _, row in flow.iterrows():
            G.add_edge(
                row["source_celltype"],
                row["target_celltype"],
                weight=row["mean_lr"],
                n_interactions=row["n_interactions"]
            )
        
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Fixed layout
        pos = nx.circular_layout(sorted(G.nodes()))
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Edges
        edges = list(G.edges())
        if len(edges) > 0:
            weights = [G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights)
            min_weight = min(weights)
            
            if max_weight > min_weight:
                edge_widths = [2 + 8 * ((w - min_weight) / (max_weight - min_weight)) for w in weights]
            else:
                edge_widths = [5] * len(weights)
            
            nx.draw_networkx_edges(
                G, pos,
                width=edge_widths,
                alpha=0.6,
                edge_color=weights,
                edge_cmap=plt.cm.plasma,
                arrows=True,
                arrowsize=20,
                arrowstyle='->',
                connectionstyle='arc3,rad=0.15',
                ax=ax,
                node_size=2500
            )
        
        # Node sizes
        node_sizes = []
        for node in sorted(G.nodes()):
            if G.degree(node) > 0 and len(edges) > 0:
                out_signal = sum([G[node][target]['weight'] for target in G.successors(node)])
                in_signal = sum([G[source][node]['weight'] for source in G.predecessors(node)])
                total = out_signal + in_signal
                node_sizes.append(800 + 150 * (total / max_weight))
            else:
                node_sizes.append(600)
        
        # Colors
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
            linewidths=2,
            ax=ax
        )
        
        # Labels
        nx.draw_networkx_labels(
            G, pos,
            labels={node: node for node in sorted(G.nodes())},
            font_size=10,
            font_weight='bold',
            font_color='black',
            ax=ax
        )
        
        ax.set_title(f"{region} Region - {age_group.capitalize()} Age\nCell Type Communication",
                     fontsize=14, fontweight='bold', pad=15)
        ax.axis('off')
        
        # Colorbar
        if len(edges) > 0 and max_weight > min_weight:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                       norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
            cbar.set_label('Mean LR Expression', fontsize=10)
        
        plt.tight_layout()
        
        outfile = os.path.join(out_dir, f"chord_{region}_{age_group}.png")
        plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved {outfile}\n")
        plt.close()

print(f"{'='*70}")
print(f"✓ All age×region chord diagrams saved to {out_dir}")
print(f"{'='*70}")
