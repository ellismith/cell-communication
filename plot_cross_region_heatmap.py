#!/usr/bin/env python3
"""
plot_cross_region_heatmap.py
----------------------------
Heatmap showing mean LR signaling (lr_means) from each source region to each target region.
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

base = "/scratch/easmit31/cell_cell/results/"
out_dir = os.path.join(base, "figures")
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(os.path.join(base, "cross_region_summary.csv"))

pivot = df.pivot(index="source_region", columns="target_region", values="lr_means")
plt.figure(figsize=(10, 8))
sns.heatmap(pivot, cmap="viridis", linewidths=0.5, annot=False)
plt.title("Average Cross-Region Signaling Strength (lr_means)", fontsize=14, fontweight="bold")
plt.xlabel("Target region")
plt.ylabel("Source region")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "cross_region_heatmap.png"), dpi=300, bbox_inches='tight')
print(f"✓ Saved cross_region_heatmap.png")
