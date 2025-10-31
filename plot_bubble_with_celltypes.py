#!/usr/bin/env python3
"""
plot_bubble_with_celltypes.py
------------------------------
Bubble plot showing cell type pairs AND regions
"""
import os, glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

base_dir = "/scratch/easmit31/cell_cell/results/pairwise/"
out_dir = os.path.join(base_dir, "figures")
os.makedirs(out_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(base_dir, "*_results.csv")))
if not files:
    raise FileNotFoundError(f"No CSVs found in {base_dir}")

print(f"Found {len(files)} result files")

rows = []
for f in files:
    df = pd.read_csv(f)
    if "lr_means" not in df.columns:
        continue
    
    # Extract ALL info from source/target
    df["source_region"] = df["source"].str.split("_").str[-1]
    df["target_region"] = df["target"].str.split("_").str[-1]
    df["source_celltype"] = df["source"].apply(lambda x: "_".join(x.split("_")[:-1]))
    df["target_celltype"] = df["target"].apply(lambda x: "_".join(x.split("_")[:-1]))
    df["interaction"] = df["ligand_complex"] + "→" + df["receptor_complex"]
    df["cell_pair"] = df["source_celltype"] + " → " + df["target_celltype"]
    
    # Within-region only
    df_within = df[df["source_region"] == df["target_region"]].copy()
    df_within["region"] = df_within["source_region"]
    
    for _, r in df_within.iterrows():
        rows.append({
            "region": r["region"],
            "cell_pair": r["cell_pair"],
            "interaction": r["interaction"],
            "lr_means": r["lr_means"],
            "magnitude_rank": r.get("magnitude_rank", 1.0)
        })

data = pd.DataFrame(rows)
print(f"Collected {len(data):,} records")
print(f"Unique cell pairs: {data['cell_pair'].nunique()}")

# Filter to significant
sig_data = data[data["magnitude_rank"] < 0.05].copy()
print(f"Significant: {len(sig_data):,}")

# Top 20 interactions
top_interactions = (sig_data.groupby("interaction")["lr_means"]
                    .mean()
                    .nlargest(20)
                    .index.tolist())

plot_data = sig_data[sig_data["interaction"].isin(top_interactions)].copy()

# Mark highly significant
plot_data["highly_sig"] = plot_data["magnitude_rank"] < 0.01

print(f"\nPlotting {len(plot_data)} points")
print(f"Cell pairs involved: {sorted(plot_data['cell_pair'].unique())}")

# Create faceted plot by cell pair
cell_pairs = sorted(plot_data['cell_pair'].unique())
n_pairs = len(cell_pairs)

# If too many cell pairs, just show top ones
if n_pairs > 6:
    top_pairs = (plot_data.groupby("cell_pair")["lr_means"]
                .mean()
                .nlargest(6)
                .index.tolist())
    plot_data = plot_data[plot_data["cell_pair"].isin(top_pairs)]
    cell_pairs = top_pairs
    n_pairs = 6

print(f"Showing top {n_pairs} cell pairs")

# Create grid
fig, axes = plt.subplots(n_pairs, 1, figsize=(14, 6*n_pairs), sharex=True)
if n_pairs == 1:
    axes = [axes]

for idx, cell_pair in enumerate(cell_pairs):
    ax = axes[idx]
    subset = plot_data[plot_data["cell_pair"] == cell_pair]
    
    # Regular
    regular = subset[~subset["highly_sig"]]
    if len(regular) > 0:
        sns.scatterplot(
            data=regular,
            x="region", y="interaction",
            size="lr_means", hue="lr_means",
            palette="viridis",
            sizes=(50, 400),
            alpha=0.6,
            ax=ax,
            legend=False
        )
    
    # Highly sig
    highly = subset[subset["highly_sig"]]
    if len(highly) > 0:
        sns.scatterplot(
            data=highly,
            x="region", y="interaction",
            size="lr_means", hue="lr_means",
            palette="viridis",
            sizes=(50, 400),
            alpha=0.8,
            edgecolor="black",
            linewidth=2,
            ax=ax
        )
    
    ax.set_title(f"{cell_pair}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Region" if idx == n_pairs-1 else "", fontsize=11)
    ax.set_ylabel("Interaction", fontsize=11)
    if idx == n_pairs-1:
        ax.tick_params(axis='x', rotation=45)

fig.suptitle("Top Interactions by Cell Type Pair Across Regions\nBlack outline = p<0.01", 
             fontsize=16, fontweight="bold", y=0.995)
plt.tight_layout()

outfile = os.path.join(out_dir, "bubble_by_celltype.png")
plt.savefig(outfile, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved to {outfile}")

# Also save summary table
summary = plot_data.groupby(["cell_pair", "region"]).agg({
    "lr_means": "mean",
    "highly_sig": "sum"
}).reset_index()
summary.to_csv(os.path.join(out_dir, "celltype_region_summary.csv"), index=False)
print(f"✓ Saved summary table")
