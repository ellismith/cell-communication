#!/usr/bin/env python3
"""
plot_sex_volcano.py
-------------------
Plot male–female differences in ligand–receptor signaling.

Assumes each CSV in /scratch/easmit31/cell_cell/results/pairwise/
includes 'sex', 'magnitude', and 'p_value' columns.

Usage:
    python plot_sex_volcano.py
"""

import os, glob, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt

base_dir = "/scratch/easmit31/cell_cell/results/pairwise/"
out_dir = os.path.join(base_dir, "figures")
os.makedirs(out_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(base_dir, "*_results.csv")))
if not files:
    raise FileNotFoundError(f"No CSVs found in {base_dir}")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

if not {"sex", "ligand", "magnitude"}.issubset(df.columns):
    raise KeyError("Input must contain columns: sex, ligand, magnitude, p_value")

# Compute average magnitude per sex per ligand
avg = df.groupby(["ligand", "sex"])["magnitude"].mean().unstack()
avg["diff"] = avg["male"] - avg["female"]

# Merge with mean p-values for visual scaling
pvals = df.groupby("ligand")["p_value"].mean()
avg = avg.join(pvals, on="ligand")

avg["neglogp"] = -np.log10(avg["p_value"])

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=avg,
    x="diff",
    y="neglogp",
    hue=(avg["diff"] > 0),
    palette={True: "steelblue", False: "salmon"},
)
plt.axvline(0, color="gray", linestyle="--")
plt.xlabel("Δ Magnitude (Male - Female)")
plt.ylabel("-log10(p)")
plt.title("Sex-Specific Signaling Differences")
plt.tight_layout()

outfile = os.path.join(out_dir, "sex_volcano.png")
plt.savefig(outfile, dpi=300)
print(f"✓ Saved volcano plot to {outfile}")
