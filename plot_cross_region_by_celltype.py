#!/usr/bin/env python3
"""
plot_cross_region_by_celltype.py
---------------------------------
Create separate bubble plots for each cell type pair showing
cross-region signaling patterns.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "/scratch/easmit31/cell_cell/results/pairwise/cross_region_analysis/"
out_dir = os.path.join(base_dir, "figures/by_celltype")
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
print(f"\nFound {len(cell_pairs)} cell type pairs:")
print(cell_pairs)

# Create one plot per cell type pair
for pair_name in cell_pairs:
    print(f"\n{'='*70}")
    print(f"Creating plot for {pair_name}")
    print(f"{'='*70}")
    
    pair_data = sig_data[sig_data["pair_name"] == pair_name].copy()
    
    if len(pair_data) == 0:
        print(f"⚠️ No data, skipping")
        continue
    
    print(f"  {len(pair_data):,} interactions")
    
    # Get top 20 interactions for this cell type pair
    top_interactions = (pair_data.groupby("interaction")["lr_means"]
                       .mean()
                       .nlargest(20)
                       .index.tolist())
    
    plot_data = pair_data[pair_data["interaction"].isin(top_interactions)].copy()
    
    # Create region pair label for x-axis
    plot_data["region_pair"] = plot_data["source_region"] + "→" + plot_data["target_region"]
    
    # Aggregate by region_pair and interaction
    agg_data = (plot_data.groupby(["region_pair", "interaction"])["lr_means"]
               .mean()
               .reset_index())
    
    print(f"  Plotting {len(agg_data)} unique points")
    
    if len(agg_data) == 0:
        print(f"⚠️ No data after aggregation, skipping")
        continue
    
    # Create bubble plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    scatter = ax.scatter(
        x=pd.Categorical(agg_data["region_pair"]).codes,
        y=pd.Categorical(agg_data["interaction"]).codes,
        s=agg_data["lr_means"] * 100,
        c=agg_data["lr_means"],
        cmap="viridis",
        alpha=0.7,
        edgecolors='none'
    )
    
    # Set labels
    region_pairs = sorted(agg_data["region_pair"].unique())
    interactions = sorted(agg_data["interaction"].unique(),
                         key=lambda x: agg_data[agg_data["interaction"]==x]["lr_means"].mean(),
                         reverse=True)
    
    ax.set_xticks(range(len(region_pairs)))
    ax.set_xticklabels(region_pairs, rotation=90, ha='right', fontsize=8)
    ax.set_yticks(range(len(interactions)))
    ax.set_yticklabels(interactions, fontsize=9)
    
    ax.set_title(f"{pair_name.replace('_', ' ↔ ')}\nCross-Region Signaling (Top 20 Interactions)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Source Region → Target Region", fontsize=12)
    ax.set_ylabel("Ligand → Receptor", fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("LR Expression", fontsize=10)
    
    plt.tight_layout()
    
    outfile = os.path.join(out_dir, f"cross_region_{pair_name}.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved {outfile}")
    plt.close()

print(f"\n{'='*70}")
print(f"✓ All cell type cross-region plots saved to {out_dir}")
print(f"{'='*70}")
