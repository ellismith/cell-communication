#!/usr/bin/env python3
"""
Violin plot of age_qval distribution for Immune/Inflammatory LR pairs
from regression output, per region.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

BASE_DIR     = "/scratch/easmit31/cell_cell/results/within_region_analysis"
ANNOTATIONS  = "/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv"
OUT_DIR      = os.path.join(BASE_DIR, "hypergeometric_all_regions")

REGION_ORDER = ["ACC", "CN", "dlPFC", "EC", "HIP", "IPP", "lCB", "M1", "MB", "mdTN", "NAc"]
REGION_LABELS = {
    "acc": "ACC", "cn": "CN", "dlpfc": "dlPFC", "ec": "EC", "hip": "HIP",
    "ipp": "IPP", "lcb": "lCB", "m1": "M1", "mb": "MB", "mdtn": "mdTN", "nac": "NAc",
}

# load annotations
ann = pd.read_csv(ANNOTATIONS)[["lr_pair","broad_category"]].drop_duplicates()
imm_pairs = set(ann[ann["broad_category"]=="Immune/Inflammatory"]["lr_pair"])

# load regression
print("Loading regression files...")
reg_dfs = []
for fpath in sorted(glob.glob(os.path.join(BASE_DIR,
                               "regression_*/whole_*_age_sex_regression.csv"))):
    region = os.path.basename(fpath).replace("whole_","").replace(
             "_age_sex_regression.csv","")
    df = pd.read_csv(fpath)
    df[["sender","receiver","ligand","receptor"]] = \
        df["interaction"].str.split("|", expand=True)
    df["lr_pair"] = df["ligand"] + "|" + df["receptor"]
    df["region"]  = region
    reg_dfs.append(df)

reg = pd.concat(reg_dfs, ignore_index=True)
reg["region_label"] = reg["region"].map(REGION_LABELS)
reg_imm = reg[reg["lr_pair"].isin(imm_pairs)].copy()

print(f"Immune/Inflammatory interactions: {len(reg_imm):,} rows across all regions")

# build data per region
data       = []
labels     = []
pct_sig    = []

for region_label in REGION_ORDER:
    sub = reg_imm[reg_imm["region_label"] == region_label]["age_qval"].dropna()
    data.append(sub.values)
    labels.append(region_label)
    pct = (sub < 0.05).mean() * 100
    pct_sig.append(pct)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# violin plot
ax = axes[0]
parts = ax.violinplot(data, positions=range(len(REGION_ORDER)),
                      showmedians=True, showextrema=True)
for pc in parts["bodies"]:
    pc.set_facecolor("steelblue")
    pc.set_alpha(0.6)
parts["cmedians"].set_color("black")
parts["cbars"].set_color("grey")
parts["cmaxes"].set_color("grey")
parts["cmins"].set_color("grey")

ax.axhline(0.05, color="#d62728", linewidth=1, linestyle="--", label="q=0.05")
ax.set_xticks(range(len(REGION_ORDER)))
ax.set_xticklabels(REGION_ORDER, fontsize=10)
ax.set_ylabel("age_qval", fontsize=11)
ax.set_title("Distribution of age q-values for Immune/Inflammatory LR pairs",
             fontsize=11)
ax.legend(fontsize=9)

# pct significant
ax2 = axes[1]
ax2.bar(range(len(REGION_ORDER)), pct_sig, color="steelblue", alpha=0.7)
ax2.axhline(5, color="#d62728", linewidth=1, linestyle="--", label="5% (expected by chance)")
ax2.set_xticks(range(len(REGION_ORDER)))
ax2.set_xticklabels(REGION_ORDER, fontsize=10)
ax2.set_ylabel("% age-sig (q<0.05)", fontsize=11)
ax2.set_title("% of Immune/Inflammatory LR pairs age-significant per region",
              fontsize=11)
ax2.legend(fontsize=9)

plt.tight_layout()
out = os.path.join(OUT_DIR, "immune_inflammatory_regression.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
