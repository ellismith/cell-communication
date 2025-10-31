#!/usr/bin/env python3
"""
plot_cross_region_age_comparison.py
------------------------------------
Compare cross-region signaling patterns across age groups.
Shows how inter-regional communication changes with age.
"""
import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = "/scratch/easmit31/cell_cell/results/pairwise_with_age/"
out_dir = os.path.join(base_dir, "figures/cross_region_age")
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
    
    # Extract info
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

# Filter to CROSS-region (source != target), same age, significant
cross = data[(data["source_age"] == data["target_age"]) & 
             (data["source_region"] != data["target_region"])].copy()
sig = cross[cross["magnitude_rank"] < 0.05].copy()
sig["age_group"] = sig["source_age"]
sig["region_pair"] = sig["source_region"] + "→" + sig["target_region"]

print(f"Significant cross-region: {len(sig):,}")

# Get age groups
age_groups = sorted(sig["age_group"].unique())
print(f"\nAge groups: {age_groups}\n")

# 1. HEATMAP: Region-to-region communication by age
print("Creating heatmaps for each age group...")
for age in age_groups:
    age_data = sig[sig["age_group"] == age].copy()
    
    # Average lr_means per region pair
    region_flow = (age_data.groupby(["source_region", "target_region"])["lr_means"]
                  .mean()
                  .reset_index())
    
    # Pivot to matrix
    matrix = region_flow.pivot(index="source_region", columns="target_region", values="lr_means")
    matrix = matrix.fillna(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(matrix, cmap="YlOrRd", annot=True, fmt=".2f", 
                cbar_kws={'label': 'Mean LR Expression'}, ax=ax)
    ax.set_title(f"Cross-Region Communication - {age.capitalize()} Age Group",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Target Region", fontsize=12)
    ax.set_ylabel("Source Region", fontsize=12)
    
    plt.tight_layout()
    outfile = os.path.join(out_dir, f"heatmap_{age}.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved {outfile}")
    plt.close()

# 2. BAR PLOT: Compare each source region's outgoing signaling across ages
print("\nCreating source region comparison plots...")
source_regions = sorted(sig["source_region"].unique())

for source_reg in source_regions:
    source_data = sig[sig["source_region"] == source_reg].copy()
    
    if len(source_data) == 0:
        continue
    
    # Average lr_means per target region per age
    flow = (source_data.groupby(["age_group", "target_region"])["lr_means"]
           .mean()
           .reset_index())
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot = flow.pivot(index="target_region", columns="age_group", values="lr_means")
    pivot = pivot.fillna(0)
    pivot.plot(kind='bar', ax=ax, width=0.8, colormap='viridis')
    
    ax.set_xlabel("Target Region", fontsize=12)
    ax.set_ylabel("Mean LR Expression", fontsize=12)
    ax.set_title(f"Outgoing Signaling from {source_reg} Across Age Groups",
                 fontsize=14, fontweight='bold')
    ax.legend(title="Age Group", fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    outfile = os.path.join(out_dir, f"outgoing_{source_reg}_by_age.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved {outfile}")
    plt.close()

# 3. SUMMARY: Total cross-region signaling by age
print("\nCreating summary comparison...")
summary = (sig.groupby(["age_group", "source_region"])["lr_means"]
          .agg(['mean', 'count'])
          .reset_index())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Mean signaling strength
pivot_mean = summary.pivot(index="source_region", columns="age_group", values="mean")
pivot_mean.plot(kind='bar', ax=axes[0], width=0.8, colormap='plasma')
axes[0].set_xlabel("Source Region", fontsize=12)
axes[0].set_ylabel("Mean LR Expression", fontsize=12)
axes[0].set_title("Average Cross-Region Signaling Strength by Age", fontsize=13, fontweight='bold')
axes[0].legend(title="Age Group")
axes[0].grid(axis='y', alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Number of interactions
pivot_count = summary.pivot(index="source_region", columns="age_group", values="count")
pivot_count.plot(kind='bar', ax=axes[1], width=0.8, colormap='plasma')
axes[1].set_xlabel("Source Region", fontsize=12)
axes[1].set_ylabel("Number of Interactions", fontsize=12)
axes[1].set_title("Cross-Region Interaction Count by Age", fontsize=13, fontweight='bold')
axes[1].legend(title="Age Group")
axes[1].grid(axis='y', alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
outfile = os.path.join(out_dir, "summary_age_comparison.png")
plt.savefig(outfile, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved {outfile}")

print(f"\n{'='*70}")
print(f"✓ All cross-region age comparison plots saved to {out_dir}")
print(f"{'='*70}")
