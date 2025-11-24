#!/usr/bin/env python3
import pandas as pd
import glob

# Load all pairwise results
result_files = glob.glob("/scratch/easmit31/cell_cell/results/pairwise/*_results.csv")
print(f"Loading {len(result_files)} files...")

data_list = []
for f in result_files:
    df = pd.read_csv(f)
    data_list.append(df)

data = pd.concat(data_list, ignore_index=True)

print(f"\nColumn names:")
print(data.columns.tolist())

# Split source and target
data[["source_cell_type", "source_region"]] = data["source"].str.rsplit("_", n=1, expand=True)
data[["target_cell_type", "target_region"]] = data["target"].str.rsplit("_", n=1, expand=True)

# Filter to significant cross-region
sig_data = data[data["qval"] < 0.05].copy()
cross_region = sig_data[sig_data["source_region"] != sig_data["target_region"]].copy()

print(f"\nTotal significant cross-region interactions: {len(cross_region)}")
print(f"\nChecking lr_means values:")
print(f"  Min: {cross_region['lr_means'].min():.6f}")
print(f"  Max: {cross_region['lr_means'].max():.6f}")
print(f"  Mean: {cross_region['lr_means'].mean():.6f}")
print(f"  Std: {cross_region['lr_means'].std():.6f}")

print(f"\nSample of cross-region interactions:")
print(cross_region.head(20))

# Check one specific cell type pair
print(f"\n\nChecking Glutamatergic → GABA:")
glu_gaba = cross_region[
    (cross_region["source_cell_type"] == "Glutamatergic") &
    (cross_region["target_cell_type"] == "GABA")
].copy()

print(f"  Total interactions: {len(glu_gaba)}")
print(f"  lr_means range: {glu_gaba['lr_means'].min():.6f} - {glu_gaba['lr_means'].max():.6f}")

# Check aggregation per region pair
print(f"\n\nAggregated by region pair (mean lr_means):")
agg = glu_gaba.groupby(['source_region', 'target_region'])['lr_means'].agg(['mean', 'count', 'std'])
print(agg.sort_values('mean', ascending=False))
