#!/usr/bin/env python3
"""
analyze_across_regions.py
-------------------------
Combine all pairwise results and extract cross-region (source_region != target_region)
ligand-receptor interactions. Summarize mean strength by region pair.
"""
import os, glob
import pandas as pd

base = "/scratch/easmit31/cell_cell/results/pairwise/"
out_dir = os.path.join(base, "cross_region_analysis")
os.makedirs(out_dir, exist_ok=True)

files = glob.glob(os.path.join(base, "*_results.csv"))
if not files:
    raise FileNotFoundError("No *_results.csv found!")

print(f"Found {len(files)} result files")

rows = []
for f in files:
    pair_name = os.path.basename(f).replace("_results.csv", "")
    df = pd.read_csv(f)
    
    if "lr_means" not in df.columns:
        print(f"⚠️ {pair_name} missing lr_means, skipping")
        continue
    
    # Extract regions from source and target columns
    df["source_region"] = df["source"].str.split("_").str[-1]
    df["target_region"] = df["target"].str.split("_").str[-1]
    df["source_celltype"] = df["source"].apply(lambda x: "_".join(x.split("_")[:-1]))
    df["target_celltype"] = df["target"].apply(lambda x: "_".join(x.split("_")[:-1]))
    
    # Create interaction identifier
    df["interaction"] = df["ligand_complex"] + "→" + df["receptor_complex"]
    
    # Keep only cross-region (where source and target are in different regions)
    across = df[df["source_region"] != df["target_region"]].copy()
    across["pair_name"] = pair_name
    
    rows.append(across)

if not rows:
    raise ValueError("No cross-region data found!")

data = pd.concat(rows, ignore_index=True)
print(f"Collected {len(data):,} cross-region interactions")

# Save detailed data
data.to_csv(os.path.join(out_dir, "cross_region_effects_detailed.csv"), index=False)
print(f"✓ Saved detailed data")

# Filter to significant only
sig_data = data[data["magnitude_rank"] < 0.05].copy()
print(f"Significant cross-region: {len(sig_data):,}")

# Summarize mean lr_means per region pair
summary = (
    sig_data.groupby(["source_region", "target_region"])
    .agg({
        "lr_means": ["mean", "median", "count"],
        "interaction": "nunique"
    })
    .reset_index()
)
summary.columns = ["source_region", "target_region", "mean_lr", "median_lr", "n_observations", "n_unique_interactions"]
summary = summary.sort_values("mean_lr", ascending=False)

summary.to_csv(os.path.join(out_dir, "cross_region_summary.csv"), index=False)
print(f"✓ Saved summary ({len(summary):,} region pairs)")

# Top cross-region signaling pairs
print("\nTop 10 cross-region signaling pairs:")
print(summary.head(10).to_string(index=False))

# Get top 20 interactions per region pair (for bubble plots)
top_interactions = []
for (src, tgt), grp in sig_data.groupby(["source_region", "target_region"]):
    # Average lr_means per interaction for this region pair
    inter_avg = grp.groupby("interaction")["lr_means"].mean().nlargest(20)
    top_20 = grp[grp["interaction"].isin(inter_avg.index)].copy()
    top_interactions.append(top_20)

top_df = pd.concat(top_interactions, ignore_index=True)
top_df.to_csv(os.path.join(out_dir, "top20_cross_region_interactions.csv"), index=False)
print(f"✓ Saved top 20 interactions per region pair ({len(top_df):,} rows)")

# Count summary
print(f"\nTotal unique region→region pairs: {len(summary)}")
print(f"Max possible (9 regions × 8 targets): 72")

