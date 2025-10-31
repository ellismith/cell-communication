#!/usr/bin/env python3
"""
plot_cross_region_bubbles.py
-----------------------------
Create bubble plots for cross-region signaling.
One plot per source region showing signaling to all target regions.
AGGREGATES multiple observations per interaction-region combo.
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

base_dir = "/scratch/easmit31/cell_cell/results/pairwise/cross_region_analysis/"
out_dir = os.path.join(base_dir, "figures")
os.makedirs(out_dir, exist_ok=True)

# Load the top 20 interactions per region pair
data_file = os.path.join(base_dir, "top20_cross_region_interactions.csv")
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Run analyze_across_regions.py first to generate {data_file}")

data = pd.read_csv(data_file)
print(f"Loaded {len(data):,} cross-region interactions")

# Get all unique source regions
source_regions = sorted(data["source_region"].unique())
print(f"Source regions: {source_regions}\n")

# Create one plot per source region
for source_reg in source_regions:
    print(f"{'='*70}")
    print(f"Creating bubble plot: {source_reg} → all other regions")
    print(f"{'='*70}")
    
    # Filter to this source region
    source_data = data[data["source_region"] == source_reg].copy()
    
    if len(source_data) == 0:
        print(f"⚠️ No data for {source_reg}, skipping\n")
        continue
    
    print(f"Found {len(source_data):,} interactions")
    
    # Get top 20 interactions across ALL target regions for this source
    top_interactions = (source_data.groupby("interaction")["lr_means"]
                       .mean()
                       .nlargest(20)
                       .index.tolist())
    
    plot_data = source_data[source_data["interaction"].isin(top_interactions)].copy()
    
    # AGGREGATE: average lr_means per interaction-target_region combo
    # This removes duplicates from different cell type pairs
    agg_data = (plot_data.groupby(["target_region", "interaction"])["lr_means"]
               .mean()
               .reset_index())
    
    print(f"Aggregated to {len(agg_data)} unique points (was {len(plot_data)})\n")
    
    # Create bubble plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    scatter = ax.scatter(
        x=pd.Categorical(agg_data["target_region"]).codes,
        y=pd.Categorical(agg_data["interaction"]).codes,
        s=agg_data["lr_means"] * 100,
        c=agg_data["lr_means"],
        cmap="viridis",
        alpha=0.7,
        edgecolors='none'
    )
    
    # Set labels
    target_regions = sorted(agg_data["target_region"].unique())
    interactions = sorted(agg_data["interaction"].unique(), 
                         key=lambda x: agg_data[agg_data["interaction"]==x]["lr_means"].mean(),
                         reverse=True)
    
    ax.set_xticks(range(len(target_regions)))
    ax.set_xticklabels(target_regions, rotation=45, ha='right')
    ax.set_yticks(range(len(interactions)))
    ax.set_yticklabels(interactions)
    
    ax.set_title(f"Cross-Region Signaling: {source_reg} → Other Regions\n" +
                 f"Top 20 Ligand-Receptor Interactions", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Target Region", fontsize=12)
    ax.set_ylabel("Ligand → Receptor", fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("LR Expression", fontsize=10)
    
    plt.tight_layout()
    
    outfile = os.path.join(out_dir, f"cross_region_bubble_{source_reg}_to_others.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {outfile}\n")
    plt.close()

print(f"{'='*70}")
print(f"✓ All cross-region bubble plots saved to {out_dir}")
print(f"{'='*70}")
