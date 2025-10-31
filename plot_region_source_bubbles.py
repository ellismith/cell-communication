#!/usr/bin/env python3
"""
plot_region_source_bubbles.py
------------------------------
For each source region, create a bubble plot showing its signaling
to all target regions (excluding self).
Creates 9 separate bubble plots.
"""
import os, glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

base_dir = "/scratch/easmit31/cell_cell/results/pairwise/"
out_dir = os.path.join(base_dir, "figures/source_region_bubbles")
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
    # Fix: use lambda to split properly
    df["source_celltype"] = df["source"].apply(lambda x: "_".join(x.split("_")[:-1]))
    df["target_celltype"] = df["target"].apply(lambda x: "_".join(x.split("_")[:-1]))
    df["interaction"] = df["ligand_complex"] + " → " + df["receptor_complex"]
    df_list.append(df)

full = pd.concat(df_list, ignore_index=True)
print(f"Total records: {len(full):,}")

# Get all unique regions
all_regions = sorted(full["source_region"].unique())
print(f"\nRegions: {all_regions}\n")

# Filter to significant interactions
sig = full[full["magnitude_rank"] < 0.05].copy()
print(f"Significant interactions: {len(sig):,}")

# For each source region
for source_reg in all_regions:
    print(f"{'='*70}")
    print(f"Creating bubble plot for {source_reg} → all other regions")
    print(f"{'='*70}")
    
    # Get data where this region is the source, but NOT the target (cross-region only)
    source_data = sig[
        (sig["source_region"] == source_reg) & 
        (sig["target_region"] != source_reg)
    ].copy()
    
    if len(source_data) == 0:
        print(f"⚠️ No cross-region signaling from {source_reg}, skipping\n")
        continue
    
    print(f"Found {len(source_data):,} cross-region interactions")
    
    # Find top N interactions by average lr_means across all target regions
    top_n = 20
    top_interactions = (source_data.groupby("interaction")["lr_means"]
                       .mean()
                       .nlargest(top_n)
                       .index.tolist())
    
    plot_data = source_data[source_data["interaction"].isin(top_interactions)].copy()
    
    # Normalize for bubble sizes
    plot_data["lr_scaled"] = plot_data["lr_means"] / plot_data["lr_means"].max()
    
    print(f"Plotting top {len(top_interactions)} interactions across {plot_data['target_region'].nunique()} target regions")
    
    # Create bubble plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.scatterplot(
        data=plot_data,
        x="target_region",
        y="interaction",
        size="lr_scaled",
        hue="lr_means",
        palette="viridis",
        sizes=(50, 400),
        alpha=0.7,
        ax=ax
    )
    
    ax.set_title(f"Top 20 Signaling Patterns from {source_reg} to Other Regions", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Target Region", fontsize=12)
    ax.set_ylabel("Ligand → Receptor", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Move legend outside
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, 
             title="LR Expression", 
             bbox_to_anchor=(1.05, 1), 
             loc="upper left", 
             fontsize=8)
    
    plt.tight_layout()
    
    outfile = os.path.join(out_dir, f"bubble_{source_reg}_to_others.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {outfile}\n")
    
    # Save data summary
    summary = plot_data.groupby(["target_region", "interaction"])["lr_means"].mean().reset_index()
    summary.to_csv(os.path.join(out_dir, f"data_{source_reg}_to_others.csv"), index=False)

print(f"{'='*70}")
print(f"✓ All bubble plots saved to {out_dir}")
print(f"{'='*70}")
