#!/usr/bin/env python3
"""
plot_summary_bubble.py
----------------------
Simple bubble plot - size = expression level, outline = highly significant
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
    
    df["source_region"] = df["source"].str.split("_").str[-1]
    df["target_region"] = df["target"].str.split("_").str[-1]
    df["interaction"] = df["ligand_complex"] + "→" + df["receptor_complex"]
    
    # Within-region only
    df_within = df[df["source_region"] == df["target_region"]].copy()
    df_within["region"] = df_within["source_region"]
    
    for _, r in df_within.iterrows():
        rows.append({
            "region": r["region"],
            "interaction": r["interaction"],
            "lr_means": r["lr_means"],
            "magnitude_rank": r.get("magnitude_rank", 1.0)
        })

data = pd.DataFrame(rows)
print(f"Collected {len(data):,} records")

# Filter to significant
sig_data = data[data["magnitude_rank"] < 0.05].copy()
print(f"Significant: {len(sig_data):,}")

# Top 20 interactions
top_interactions = (sig_data.groupby("interaction")["lr_means"]
                    .mean()
                    .nlargest(20)
                    .index.tolist())

plot_data = sig_data[sig_data["interaction"].isin(top_interactions)].copy()

# Mark highly significant for outline
plot_data["highly_sig"] = plot_data["magnitude_rank"] < 0.01

# Pivot to get all region combos
pivot = plot_data.pivot_table(
    index="interaction",
    columns="region",
    values="lr_means",
    aggfunc="mean"
)

highly_pivot = plot_data.pivot_table(
    index="interaction", 
    columns="region",
    values="highly_sig",
    aggfunc="any"
)

# Melt back
plot_full = pivot.reset_index().melt(id_vars="interaction", var_name="region", value_name="lr_means").dropna()
highly_full = highly_pivot.reset_index().melt(id_vars="interaction", var_name="region", value_name="highly_sig").dropna()
plot_full = plot_full.merge(highly_full, on=["interaction", "region"], how="left")
plot_full["highly_sig"] = plot_full["highly_sig"].fillna(False)

print(f"\nPlotting {len(plot_full)} points ({plot_full['highly_sig'].sum()} with outline)")

# Plot
fig, ax = plt.subplots(figsize=(14, 10))

# Regular (no outline)
regular = plot_full[~plot_full["highly_sig"]]
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

# Highly sig (black outline)
highly = plot_full[plot_full["highly_sig"]]
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

ax.set_title("Top 20 Interactions Across Regions\nBlack outline = p<0.01", fontsize=14, fontweight="bold")
ax.set_xlabel("Region", fontsize=12)
ax.set_ylabel("Ligand → Receptor", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title="LR Expression", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()

plt.savefig(os.path.join(out_dir, "summary_bubble_plot.png"), dpi=300, bbox_inches='tight')
print(f"✓ Saved!")
