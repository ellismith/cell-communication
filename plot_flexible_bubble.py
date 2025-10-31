#!/usr/bin/env python3
"""
plot_flexible_bubble.py
-----------------------
Flexible bubble plot for any pairwise results directory.
Usage: python plot_flexible_bubble.py <results_dir> <output_suffix>
Example: python plot_flexible_bubble.py pairwise_with_age age
"""
import os, glob, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("Usage: python plot_flexible_bubble.py <results_dir> <output_suffix>")
    print("Example: python plot_flexible_bubble.py pairwise_with_age age")
    sys.exit(1)

results_subdir = sys.argv[1]  # e.g., "pairwise_with_age"
output_suffix = sys.argv[2]   # e.g., "age"

base_dir = f"/scratch/easmit31/cell_cell/results/{results_subdir}/"
out_dir = os.path.join(base_dir, "figures")
os.makedirs(out_dir, exist_ok=True)

# Gather CSVs
files = sorted(glob.glob(os.path.join(base_dir, "*_results.csv")))
if not files:
    raise FileNotFoundError(f"No CSVs found in {base_dir}")

print(f"Found {len(files)} result files in {results_subdir}")

rows = []
for f in files:
    df = pd.read_csv(f)
    
    if "lr_means" not in df.columns:
        print(f"⚠️ Skipping {f} (no lr_means column)")
        continue
    
    # Parse the grouping from source/target columns
    # Format could be: CellType_Region or CellType_Region_Age or CellType_Region_Sex
    df["interaction"] = df["ligand_complex"] + " → " + df["receptor_complex"]
    
    for _, r in df.iterrows():
        rows.append({
            "source": r["source"],
            "target": r["target"],
            "interaction": r["interaction"],
            "lr_means": r["lr_means"],
            "magnitude_rank": r.get("magnitude_rank", 1.0)
        })

data = pd.DataFrame(rows)
if data.empty:
    raise ValueError("No valid data found.")

print(f"Collected {len(data):,} interaction records")

# Filter to significant interactions
sig_data = data[data["magnitude_rank"] < 0.05].copy()
print(f"Significant interactions: {len(sig_data):,}")

# Take top N by lr_means
top_n = 20
if len(sig_data) > top_n:
    top_interactions = (sig_data.groupby("interaction")["lr_means"]
                       .mean()
                       .nlargest(top_n)
                       .index.tolist())
    sig_data = sig_data[sig_data["interaction"].isin(top_interactions)]
    print(f"Showing top {top_n} interactions")

# Normalize for bubble sizes
sig_data["lr_scaled"] = sig_data["lr_means"] / sig_data["lr_means"].max()

# For x-axis, use source groups (could be region, region_age, or region_sex)
sig_data["source_group"] = sig_data["source"]
unique_groups = sorted(sig_data["source_group"].unique())
print(f"Unique source groups: {len(unique_groups)}")

# Create plot
fig, ax = plt.subplots(figsize=(16, 12))
sns.scatterplot(
    data=sig_data,
    x="source_group",
    y="interaction",
    size="lr_scaled",
    hue="lr_means",
    palette="viridis",
    sizes=(50, 400),
    alpha=0.7,
    ax=ax
)

ax.set_title(f"Top {top_n} Ligand-Receptor Interactions ({output_suffix.title()} Stratified)", 
             fontsize=14, fontweight='bold')
ax.set_xlabel(f"Source Group ({output_suffix})", fontsize=12)
ax.set_ylabel("Ligand → Receptor", fontsize=12)
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.legend(title="LR Expression", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()

outfile = os.path.join(out_dir, f"bubble_plot_{output_suffix}.png")
plt.savefig(outfile, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved bubble plot to {outfile}")

# Save summary
summary = sig_data.groupby("source_group").agg({
    "lr_means": ["mean", "median", "count"]
}).round(3)
summary.to_csv(os.path.join(out_dir, f"summary_{output_suffix}.csv"))
print(f"✓ Saved summary to {out_dir}/summary_{output_suffix}.csv")
