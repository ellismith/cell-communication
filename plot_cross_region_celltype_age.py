#!/usr/bin/env python3
"""
plot_cross_region_celltype_age.py
----------------------------------
Show cell type communication ACROSS regions by age group.
E.g., Astrocyte_EC → GABA_dlPFC
"""
import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = "/scratch/easmit31/cell_cell/results/pairwise_with_age/"
out_dir = os.path.join(base_dir, "figures/cross_region_celltype_age")
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
    df["source_celltype"] = df["source"].apply(lambda x: "_".join(x.split("_")[:-2]))
    df["target_celltype"] = df["target"].apply(lambda x: "_".join(x.split("_")[:-2]))
    df["interaction"] = df["ligand_complex"] + "→" + df["receptor_complex"]
    
    # Create cell-region pairs
    df["source_cell_region"] = df["source_celltype"] + "_" + df["source_region"]
    df["target_cell_region"] = df["target_celltype"] + "_" + df["target_region"]
    df["cell_region_pair"] = df["source_cell_region"] + " → " + df["target_cell_region"]
    
    rows.append(df)

data = pd.concat(rows, ignore_index=True)
print(f"Total records: {len(data):,}")

# Filter to CROSS-REGION, same age, significant
cross = data[(data["source_age"] == data["target_age"]) & 
             (data["source_region"] != data["target_region"])].copy()
sig = cross[cross["magnitude_rank"] < 0.05].copy()
sig["age_group"] = sig["source_age"]

print(f"Significant cross-region: {len(sig):,}")

# Age order
age_order = ['young', 'medium', 'old']
age_groups = [age for age in age_order if age in sig["age_group"].unique()]
print(f"\nAge groups: {age_groups}\n")

# 1. HEATMAP: Top cell-region pairs by age
print("Creating heatmap of top cross-region cell type interactions...")

# Get top 30 cell-region pairs overall
top_pairs = (sig.groupby("cell_region_pair")["lr_means"]
            .mean()
            .nlargest(30)
            .index.tolist())

plot_data = sig[sig["cell_region_pair"].isin(top_pairs)].copy()

# Aggregate by age and pair
agg = (plot_data.groupby(["age_group", "cell_region_pair"])["lr_means"]
      .mean()
      .reset_index())

# Pivot
pivot = agg.pivot(index="cell_region_pair", columns="age_group", values="lr_means")
pivot = pivot[[age for age in age_order if age in pivot.columns]]  # Fix column order
pivot = pivot.fillna(0)

# Sort by mean
pivot["mean"] = pivot.mean(axis=1)
pivot = pivot.sort_values("mean", ascending=False).drop("mean", axis=1)

# Plot
fig, ax = plt.subplots(figsize=(10, 14))
sns.heatmap(pivot, cmap="YlOrRd", annot=False, fmt=".2f",
           cbar_kws={'label': 'Mean LR Expression'}, ax=ax)
ax.set_title("Top 30 Cross-Region Cell Type Interactions Across Age",
            fontsize=14, fontweight='bold')
ax.set_xlabel("Age Group", fontsize=12)
ax.set_ylabel("Source Cell_Region → Target Cell_Region", fontsize=12)

plt.tight_layout()
outfile = os.path.join(out_dir, "heatmap_celltype_pairs_by_age.png")
plt.savefig(outfile, dpi=300, bbox_inches='tight')
print(f"✓ Saved {outfile}\n")
plt.close()

# 2. BAR PLOTS: For each major cell type pair, show cross-region patterns by age
print("Creating cell type pair comparison plots...")

# Get ALL cell type pairs
sig["celltype_pair"] = sig["source_celltype"] + " → " + sig["target_celltype"]
all_celltype_pairs = sorted(sig["celltype_pair"].unique())
print(f"Found {len(all_celltype_pairs)} cell type pairs\n")

for ct_pair in all_celltype_pairs:
    pair_data = sig[sig["celltype_pair"] == ct_pair].copy()
    
    # Aggregate by age and region pair
    agg = (pair_data.groupby(["age_group", "source_region", "target_region"])["lr_means"]
          .mean()
          .reset_index())
    agg["region_pair"] = agg["source_region"] + "→" + agg["target_region"]
    
    # Get top 20 region pairs for this cell type pair
    top_reg_pairs = (agg.groupby("region_pair")["lr_means"]
                    .mean()
                    .nlargest(20)
                    .index.tolist())
    
    plot_agg = agg[agg["region_pair"].isin(top_reg_pairs)]
    
    if len(plot_agg) < 3:  # Need at least a few data points
        print(f"  Skipping {ct_pair} (insufficient data)")
        continue
    
    # Pivot
    pivot = plot_agg.pivot(index="region_pair", columns="age_group", values="lr_means")
    pivot = pivot[[age for age in age_order if age in pivot.columns]]  # Fix order
    pivot = pivot.fillna(0)
    
    # Sort
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=False).drop("mean", axis=1)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    pivot.plot(kind='barh', ax=ax, width=0.8, colormap='viridis')
    
    ax.set_xlabel("Mean LR Expression", fontsize=12)
    ax.set_ylabel("Source Region → Target Region", fontsize=12)
    ax.set_title(f"{ct_pair}\nCross-Region Signaling by Age (Top 20 Region Pairs)",
                fontsize=14, fontweight='bold')
    ax.legend(title="Age Group", fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Safe filename
    safe_name = ct_pair.replace(" ", "_").replace("→", "to")
    outfile = os.path.join(out_dir, f"barplot_{safe_name}_by_age.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved {outfile}")
    plt.close()

print(f"\n{'='*70}")
print(f"✓ All cross-region cell type age plots saved to {out_dir}")
print(f"{'='*70}")
