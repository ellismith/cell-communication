#!/usr/bin/env python3
"""
plot_each_region_vs_all.py
---------------------------
For each region, create a volcano plot comparing its signaling
to the average of all other regions.
Creates 9 separate plots (one per region).
"""
import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base_dir = "/scratch/easmit31/cell_cell/results/pairwise/"
out_dir = os.path.join(base_dir, "figures/region_specific")
os.makedirs(out_dir, exist_ok=True)

# Load all results
files = sorted(glob.glob(os.path.join(base_dir, "*_results.csv")))
if not files:
    raise FileNotFoundError(f"No CSVs found in {base_dir}")

print(f"Loading {len(files)} result files...")
df_list = []
for f in files:
    df = pd.read_csv(f)
    df["source_region"] = df["source"].str.split("_").str[-1]
    df["target_region"] = df["target"].str.split("_").str[-1]
    df["interaction"] = df["ligand_complex"] + " → " + df["receptor_complex"]
    df_list.append(df)

full = pd.concat(df_list, ignore_index=True)
print(f"Total records: {len(full):,}")

# Get all unique regions
all_regions = sorted(full["source_region"].unique())
print(f"\nRegions: {all_regions}\n")

# For each region, compare to all others
for focal_region in all_regions:
    print(f"{'='*70}")
    print(f"Analyzing {focal_region} vs all other regions")
    print(f"{'='*70}")
    
    # Get interactions involving focal region (as source OR target)
    focal_data = full[
        (full["source_region"] == focal_region) | 
        (full["target_region"] == focal_region)
    ].copy()
    
    # Get interactions from all OTHER regions
    other_data = full[
        ((full["source_region"] != focal_region) & (full["target_region"] != focal_region))
    ].copy()
    
    print(f"{focal_region} interactions: {len(focal_data):,}")
    print(f"Other regions interactions: {len(other_data):,}")
    
    # Average lr_means per interaction
    focal_avg = focal_data.groupby("interaction")["lr_means"].mean()
    other_avg = other_data.groupby("interaction")["lr_means"].mean()
    
    # Find common interactions
    common = focal_avg.index.intersection(other_avg.index)
    print(f"Common interactions: {len(common)}")
    
    if len(common) < 10:
        print(f"⚠️ Too few common interactions for {focal_region}, skipping\n")
        continue
    
    # Calculate differences
    compare = pd.DataFrame({
        f"{focal_region}": focal_avg[common],
        "other_regions": other_avg[common],
    })
    compare["diff"] = compare[focal_region] - compare["other_regions"]
    compare["log_fc"] = np.log2((compare[focal_region] + 0.01) / (compare["other_regions"] + 0.01))
    
    # Get significance (average magnitude_rank)
    focal_rank = focal_data.groupby("interaction")["magnitude_rank"].mean()
    other_rank = other_data.groupby("interaction")["magnitude_rank"].mean()
    compare["focal_rank"] = focal_rank[common]
    compare["other_rank"] = other_rank[common]
    compare["avg_rank"] = (compare["focal_rank"] + compare["other_rank"]) / 2
    compare["neglog_rank"] = -np.log10(compare["avg_rank"] + 1e-10)
    
    # Determine significance
    diff_threshold = 0.1
    rank_threshold = 0.05
    
    compare["category"] = "Not significant"
    compare.loc[
        (compare["diff"] > diff_threshold) & (compare["avg_rank"] < rank_threshold),
        "category"
    ] = f"{focal_region} enriched"
    compare.loc[
        (compare["diff"] < -diff_threshold) & (compare["avg_rank"] < rank_threshold),
        "category"
    ] = "Other regions enriched"
    
    n_enriched = (compare["category"] == f"{focal_region} enriched").sum()
    n_depleted = (compare["category"] == "Other regions enriched").sum()
    
    print(f"{focal_region} enriched: {n_enriched}")
    print(f"Other regions enriched: {n_depleted}\n")
    
    # Create volcano plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot by category
    colors = {
        "Not significant": "lightgray",
        f"{focal_region} enriched": "firebrick",
        "Other regions enriched": "steelblue"
    }
    
    for cat in compare["category"].unique():
        subset = compare[compare["category"] == cat]
        ax.scatter(
            subset["diff"],
            subset["neglog_rank"],
            c=colors[cat],
            label=cat,
            s=30 if cat == "Not significant" else 60,
            alpha=0.5 if cat == "Not significant" else 0.7,
            edgecolors="black" if cat != "Not significant" else "none",
            linewidth=0.5
        )
    
    # Add threshold lines
    ax.axvline(diff_threshold, color="gray", linestyle="--", linewidth=1, alpha=0.3)
    ax.axvline(-diff_threshold, color="gray", linestyle="--", linewidth=1, alpha=0.3)
    ax.axhline(-np.log10(rank_threshold), color="red", linestyle="--", linewidth=1, alpha=0.3)
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
    
    # Label top interactions
    enriched = compare[compare["category"] == f"{focal_region} enriched"].nlargest(5, "diff")
    depleted = compare[compare["category"] == "Other regions enriched"].nsmallest(5, "diff")
    
    for idx in enriched.index:
        ax.annotate(
            idx.split("→")[0][:20],  # Just ligand, truncated
            (compare.loc[idx, "diff"], compare.loc[idx, "neglog_rank"]),
            fontsize=7,
            alpha=0.8,
            xytext=(5, 5),
            textcoords="offset points"
        )
    
    for idx in depleted.index:
        ax.annotate(
            idx.split("→")[0][:20],
            (compare.loc[idx, "diff"], compare.loc[idx, "neglog_rank"]),
            fontsize=7,
            alpha=0.8,
            xytext=(-5, -5),
            textcoords="offset points",
            ha="right"
        )
    
    ax.set_xlabel(f"Δ LR Expression ({focal_region} - Other Regions)", fontsize=12)
    ax.set_ylabel("-log10(magnitude rank)", fontsize=12)
    ax.set_title(f"{focal_region} vs All Other Regions\n{n_enriched} enriched, {n_depleted} depleted", 
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    outfile = os.path.join(out_dir, f"volcano_{focal_region}_vs_others.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {outfile}\n")
    
    # Save data
    compare_sorted = compare.sort_values("diff", ascending=False)
    compare_sorted.to_csv(os.path.join(out_dir, f"data_{focal_region}_vs_others.csv"))

print(f"\n{'='*70}")
print(f"✓ All volcano plots saved to {out_dir}")
print(f"{'='*70}")
