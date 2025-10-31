#!/usr/bin/env python3
"""
plot_region_volcano.py
----------------------
Plot differences in ligand-receptor signaling between two regions.
Uses lr_means as the magnitude metric.
Compares within-region interactions (e.g., EC→EC vs dlPFC→dlPFC).
"""
import os, glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

base_dir = "/scratch/easmit31/cell_cell/results/pairwise/"
out_dir = os.path.join(base_dir, "figures")
os.makedirs(out_dir, exist_ok=True)

# Load all results
files = sorted(glob.glob(os.path.join(base_dir, "*_results.csv")))
if not files:
    raise FileNotFoundError(f"No CSVs found in {base_dir}")

print(f"Loading {len(files)} result files...")
df_list = []
for f in files:
    df = pd.read_csv(f)
    # Extract regions from source/target
    df["source_region"] = df["source"].str.split("_").str[-1]
    df["target_region"] = df["target"].str.split("_").str[-1]
    df["interaction"] = df["ligand_complex"] + " → " + df["receptor_complex"]
    df_list.append(df)

full = pd.concat(df_list, ignore_index=True)
print(f"Total records: {len(full):,}")

# Compare two specific regions (you can change these)
region1 = "EC"
region2 = "dlPFC"

print(f"\nComparing {region1} vs {region2}...")

# Filter to same-region interactions
df1 = full[(full["source_region"] == region1) & (full["target_region"] == region1)].copy()
df2 = full[(full["source_region"] == region2) & (full["target_region"] == region2)].copy()

print(f"{region1} interactions: {len(df1):,}")
print(f"{region2} interactions: {len(df2):,}")

# Average lr_means per interaction for each region
avg1 = df1.groupby("interaction")["lr_means"].mean()
avg2 = df2.groupby("interaction")["lr_means"].mean()

# Find common interactions
common = avg1.index.intersection(avg2.index)
print(f"Common interactions: {len(common)}")

# Calculate difference
compare = pd.DataFrame({
    f"{region1}_mean": avg1[common],
    f"{region2}_mean": avg2[common],
})
compare["diff"] = compare[f"{region1}_mean"] - compare[f"{region2}_mean"]
compare["log_fold_change"] = np.log2((compare[f"{region1}_mean"] + 0.01) / (compare[f"{region2}_mean"] + 0.01))

# Get magnitude_rank as proxy for significance
rank1 = df1.groupby("interaction")["magnitude_rank"].mean()
rank2 = df2.groupby("interaction")["magnitude_rank"].mean()
compare["avg_rank"] = (rank1[common] + rank2[common]) / 2
compare["neglog_rank"] = -np.log10(compare["avg_rank"] + 1e-10)

# Create volcano plot
fig, ax = plt.subplots(figsize=(10, 8))

# Color by significance and direction
compare["color"] = "gray"
compare.loc[(compare["diff"] > 0.1) & (compare["avg_rank"] < 0.05), "color"] = f"{region1} enriched"
compare.loc[(compare["diff"] < -0.1) & (compare["avg_rank"] < 0.05), "color"] = f"{region2} enriched"

for color in compare["color"].unique():
    subset = compare[compare["color"] == color]
    if color == "gray":
        ax.scatter(subset["diff"], subset["neglog_rank"], 
                  c="lightgray", alpha=0.5, s=20, label="Not significant")
    elif region1 in color:
        ax.scatter(subset["diff"], subset["neglog_rank"],
                  c="steelblue", alpha=0.7, s=50, label=color)
    else:
        ax.scatter(subset["diff"], subset["neglog_rank"],
                  c="salmon", alpha=0.7, s=50, label=color)

ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1, label="p=0.05 threshold")

ax.set_xlabel(f"Δ LR Expression ({region1} - {region2})", fontsize=12)
ax.set_ylabel("-log10(magnitude rank)", fontsize=12)
ax.set_title(f"Region-Specific Signaling: {region1} vs {region2}", fontsize=14)
ax.legend()
plt.tight_layout()

outfile = os.path.join(out_dir, f"volcano_{region1}_vs_{region2}.png")
plt.savefig(outfile, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved volcano plot to {outfile}")

# Save top differences
top_diff = compare.nlargest(20, "diff")
top_diff.to_csv(os.path.join(out_dir, f"top_{region1}_enriched.csv"))
print(f"✓ Saved top {region1}-enriched interactions")

top_diff2 = compare.nsmallest(20, "diff")
top_diff2.to_csv(os.path.join(out_dir, f"top_{region2}_enriched.csv"))
print(f"✓ Saved top {region2}-enriched interactions")
