#!/usr/bin/env python3
"""
plot_cross_region_sex_comparison.py
------------------------------------
Compare cross-region signaling patterns between sexes.
Shows how inter-regional communication differs by sex.
"""
import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = "/scratch/easmit31/cell_cell/results/pairwise_with_sex/"
out_dir = os.path.join(base_dir, "figures/cross_region_sex")
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
    
    # Extract info
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

# Filter to CROSS-region, same sex, significant
cross = data[(data["source_sex"] == data["target_sex"]) & 
             (data["source_region"] != data["target_region"])].copy()
sig = cross[cross["magnitude_rank"] < 0.05].copy()
sig["sex_group"] = sig["source_sex"]
sig["region_pair"] = sig["source_region"] + "→" + sig["target_region"]

print(f"Significant cross-region: {len(sig):,}")

# Get sex groups
sex_groups = sorted(sig["sex_group"].unique())
print(f"\nSex groups: {sex_groups}\n")

# 1. HEATMAP: Region-to-region communication by sex
print("Creating heatmaps for each sex...")
for sex in sex_groups:
    sex_data = sig[sig["sex_group"] == sex].copy()
    
    # Average lr_means per region pair
    region_flow = (sex_data.groupby(["source_region", "target_region"])["lr_means"]
                  .mean()
                  .reset_index())
    
    # Pivot to matrix
    matrix = region_flow.pivot(index="source_region", columns="target_region", values="lr_means")
    matrix = matrix.fillna(0)
    
    # Plot
    sex_label = "Male" if sex == "M" else "Female"
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(matrix, cmap="YlOrRd", annot=True, fmt=".2f", 
                cbar_kws={'label': 'Mean LR Expression'}, ax=ax)
    ax.set_title(f"Cross-Region Communication - {sex_label}",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Target Region", fontsize=12)
    ax.set_ylabel("Source Region", fontsize=12)
    
    plt.tight_layout()
    outfile = os.path.join(out_dir, f"heatmap_{sex}.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved {outfile}")
    plt.close()

# 2. BAR PLOT: Compare each source region's outgoing signaling by sex
print("\nCreating source region comparison plots...")
source_regions = sorted(sig["source_region"].unique())

for source_reg in source_regions:
    source_data = sig[sig["source_region"] == source_reg].copy()
    
    if len(source_data) == 0:
        continue
    
    # Average lr_means per target region per sex
    flow = (source_data.groupby(["sex_group", "target_region"])["lr_means"]
           .mean()
           .reset_index())
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot = flow.pivot(index="target_region", columns="sex_group", values="lr_means")
    pivot = pivot.fillna(0)
    pivot.columns = ["Female" if c == "F" else "Male" for c in pivot.columns]
    pivot.plot(kind='bar', ax=ax, width=0.8, color=['#FF6B9D', '#4A90E2'])
    
    ax.set_xlabel("Target Region", fontsize=12)
    ax.set_ylabel("Mean LR Expression", fontsize=12)
    ax.set_title(f"Outgoing Signaling from {source_reg}: Male vs Female",
                 fontsize=14, fontweight='bold')
    ax.legend(title="Sex", fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    outfile = os.path.join(out_dir, f"outgoing_{source_reg}_by_sex.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved {outfile}")
    plt.close()

# 3. SUMMARY: Total cross-region signaling by sex
print("\nCreating summary comparison...")
summary = (sig.groupby(["sex_group", "source_region"])["lr_means"]
          .agg(['mean', 'count'])
          .reset_index())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Mean signaling strength
pivot_mean = summary.pivot(index="source_region", columns="sex_group", values="mean")
pivot_mean.columns = ["Female" if c == "F" else "Male" for c in pivot_mean.columns]
pivot_mean.plot(kind='bar', ax=axes[0], width=0.8, color=['#FF6B9D', '#4A90E2'])
axes[0].set_xlabel("Source Region", fontsize=12)
axes[0].set_ylabel("Mean LR Expression", fontsize=12)
axes[0].set_title("Average Cross-Region Signaling Strength by Sex", fontsize=13, fontweight='bold')
axes[0].legend(title="Sex")
axes[0].grid(axis='y', alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Number of interactions
pivot_count = summary.pivot(index="source_region", columns="sex_group", values="count")
pivot_count.columns = ["Female" if c == "F" else "Male" for c in pivot_count.columns]
pivot_count.plot(kind='bar', ax=axes[1], width=0.8, color=['#FF6B9D', '#4A90E2'])
axes[1].set_xlabel("Source Region", fontsize=12)
axes[1].set_ylabel("Number of Interactions", fontsize=12)
axes[1].set_title("Cross-Region Interaction Count by Sex", fontsize=13, fontweight='bold')
axes[1].legend(title="Sex")
axes[1].grid(axis='y', alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
outfile = os.path.join(out_dir, "summary_sex_comparison.png")
plt.savefig(outfile, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved {outfile}")

print(f"\n{'='*70}")
print(f"✓ All cross-region sex comparison plots saved to {out_dir}")
print(f"{'='*70}")
