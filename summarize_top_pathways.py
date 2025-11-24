#!/usr/bin/env python3
"""
Summarize top pathways for each cell type pair
Creates tables showing which pathways are driving the communication
"""
import pandas as pd
import glob
import os

base_dir = "/scratch/easmit31/cell_cell/results/pairwise/"
out_dir = os.path.join(base_dir, "pathway_summaries")
os.makedirs(out_dir, exist_ok=True)

# Load all results
files = glob.glob(base_dir + "*_results.csv")
all_data = []

for f in files:
    df = pd.read_csv(f)
    df["source_region"] = df["source"].str.split("_").str[-1]
    df["target_region"] = df["target"].str.split("_").str[-1]
    df["source_celltype"] = df["source"].apply(lambda x: "_".join(x.split("_")[:-1]))
    df["target_celltype"] = df["target"].apply(lambda x: "_".join(x.split("_")[:-1]))
    df["interaction"] = df["ligand_complex"] + " → " + df["receptor_complex"]
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# Get significant interactions
sig = data[data["qval"] < 0.05].copy()
sig["cell_pair"] = sig["source_celltype"] + " → " + sig["target_celltype"]

# Within-region
within = sig[sig["source_region"] == sig["target_region"]].copy()
within["region"] = within["source_region"]

# Cross-region
cross = sig[sig["source_region"] != sig["target_region"]].copy()

print(f"Significant interactions: {len(sig):,}")
print(f"  Within-region: {len(within):,}")
print(f"  Cross-region: {len(cross):,}\n")

# 1. TOP PATHWAYS PER CELL TYPE PAIR (WITHIN-REGION)
print("="*70)
print("TOP PATHWAYS FOR EACH CELL TYPE PAIR (WITHIN-REGION)")
print("="*70)

within_summary = []
for cell_pair in sorted(within["cell_pair"].unique()):
    pair_data = within[within["cell_pair"] == cell_pair]
    
    # Get top 10 pathways by mean lr_means
    top_pathways = (pair_data.groupby("interaction")
                   .agg({"lr_means": "mean", "qval": "min"})
                   .sort_values("lr_means", ascending=False)
                   .head(10))
    
    for pathway, row in top_pathways.iterrows():
        within_summary.append({
            "cell_pair": cell_pair,
            "pathway": pathway,
            "mean_lr": row["lr_means"],
            "min_qval": row["qval"]
        })

within_df = pd.DataFrame(within_summary)
within_file = os.path.join(out_dir, "top_pathways_within_region.csv")
within_df.to_csv(within_file, index=False)
print(f"✓ Saved: {within_file}\n")

# Print preview
for cell_pair in sorted(within["cell_pair"].unique())[:3]:
    print(f"\n{cell_pair}:")
    pair_pathways = within_df[within_df["cell_pair"] == cell_pair].head(5)
    for _, row in pair_pathways.iterrows():
        print(f"  {row['pathway']:50s} | lr_means={row['mean_lr']:.2f} | q={row['min_qval']:.2e}")

# 2. TOP PATHWAYS PER CELL TYPE PAIR (CROSS-REGION)
print("\n" + "="*70)
print("TOP PATHWAYS FOR EACH CELL TYPE PAIR (CROSS-REGION)")
print("="*70)

cross_summary = []
for cell_pair in sorted(cross["cell_pair"].unique()):
    pair_data = cross[cross["cell_pair"] == cell_pair]
    
    # Get top 10 pathways by mean lr_means
    top_pathways = (pair_data.groupby("interaction")
                   .agg({"lr_means": "mean", "qval": "min"})
                   .sort_values("lr_means", ascending=False)
                   .head(10))
    
    for pathway, row in top_pathways.iterrows():
        cross_summary.append({
            "cell_pair": cell_pair,
            "pathway": pathway,
            "mean_lr": row["lr_means"],
            "min_qval": row["qval"]
        })

cross_df = pd.DataFrame(cross_summary)
cross_file = os.path.join(out_dir, "top_pathways_cross_region.csv")
cross_df.to_csv(cross_file, index=False)
print(f"✓ Saved: {cross_file}\n")

# Print preview
for cell_pair in sorted(cross["cell_pair"].unique())[:3]:
    print(f"\n{cell_pair}:")
    pair_pathways = cross_df[cross_df["cell_pair"] == cell_pair].head(5)
    for _, row in pair_pathways.iterrows():
        print(f"  {row['pathway']:50s} | lr_means={row['mean_lr']:.2f} | q={row['min_qval']:.2e}")

print("\n" + "="*70)
print(f"✓ All pathway summaries saved to {out_dir}")
print("="*70)
