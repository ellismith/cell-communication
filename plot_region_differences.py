#!/usr/bin/env python3
"""
plot_region_differences.py
--------------------------
Compare ligand-receptor signaling magnitude (lr_means) across regions.
The source/target columns already contain region info like 'Astrocyte_EC'
"""
import os, glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
base_dir = "/scratch/easmit31/cell_cell/results/pairwise/"
out_dir = os.path.join(base_dir, "figures")
os.makedirs(out_dir, exist_ok=True)

# Gather CSVs
files = sorted(glob.glob(os.path.join(base_dir, "*_results.csv")))
if not files:
    raise FileNotFoundError(f"No CSVs found in {base_dir}")

print(f"Found {len(files)} result files")

df_list = []
for f in files:
    df = pd.read_csv(f)
    name = os.path.basename(f).replace("_results.csv", "")
    
    # Check required columns
    if "lr_means" not in df.columns:
        print(f"⚠️ 'lr_means' not found in {name}, skipping")
        continue
    
    df["pair"] = name
    df_list.append(df)

if not df_list:
    raise ValueError("No valid files with 'lr_means' found.")

# Combine all results
full = pd.concat(df_list, ignore_index=True)
print(f"Combined data: {len(full):,} rows")

# Extract region from source/target (they already have format like 'Astrocyte_EC')
full["source_region"] = full["source"].str.split("_").str[-1]
full["target_region"] = full["target"].str.split("_").str[-1]
full["region_pair"] = full["source_region"] + "→" + full["target_region"]

# Create interaction label
full["interaction"] = full["ligand_complex"] + " → " + full["receptor_complex"]

print(f"\nRegion pairs found: {sorted(full['region_pair'].unique())}")
print(f"Total interactions: {len(full['interaction'].unique())}")

# Pivot: average lr_means per interaction × region pair
pivot = full.pivot_table(
    values="lr_means", 
    index="interaction", 
    columns="region_pair", 
    aggfunc="mean"
)

print(f"\nPivot table: {pivot.shape[0]} interactions × {pivot.shape[1]} region pairs")

# Plot heatmap
fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(
    pivot, 
    cmap="coolwarm", 
    center=0,
    cbar_kws={'label': 'Mean LR Expression'},
    ax=ax
)
ax.set_title("Regional Differences in Ligand-Receptor Signaling", fontsize=14)
ax.set_xlabel("Region Pair (Source → Target)", fontsize=12)
ax.set_ylabel("Ligand → Receptor", fontsize=12)
plt.tight_layout()

outfile = os.path.join(out_dir, "region_differences_heatmap.png")
plt.savefig(outfile, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved regional heatmap to {outfile}")

# Also save top varying interactions
variance = pivot.var(axis=1).sort_values(ascending=False)
print(f"\nTop 10 most region-variable interactions:")
print(variance.head(10))

variance.head(20).to_csv(os.path.join(out_dir, "top_variable_interactions.csv"))
print(f"✓ Saved top variable interactions to {out_dir}/top_variable_interactions.csv")
