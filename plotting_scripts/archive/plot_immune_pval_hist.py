#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

BASE_DIR    = "/scratch/easmit31/cell_cell/results/within_region_analysis"
ANNOTATIONS = "/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv"
OUT_DIR     = os.path.join(BASE_DIR, "hypergeometric_all_regions")

REGION_ORDER = ["ACC", "CN", "dlPFC", "EC", "HIP", "IPP", "lCB", "M1", "MB", "mdTN", "NAc"]
REGION_LABELS = {
    "acc": "ACC", "cn": "CN", "dlpfc": "dlPFC", "ec": "EC", "hip": "HIP",
    "ipp": "IPP", "lcb": "lCB", "m1": "M1", "mb": "MB", "mdtn": "mdTN", "nac": "NAc",
}

ann = pd.read_csv(ANNOTATIONS)[["lr_pair","broad_category"]].drop_duplicates()
imm_pairs = set(ann[ann["broad_category"]=="Immune/Inflammatory"]["lr_pair"])

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

fig, axes = plt.subplots(3, 4, figsize=(14, 10), sharey=False)
axes = axes.flatten()

for i, region_label in enumerate(REGION_ORDER):
    ax  = axes[i]
    sub = reg_imm[reg_imm["region_label"] == region_label]["age_pval"].dropna()
    ax.hist(sub, bins=20, range=(0,1), color="steelblue", alpha=0.7, edgecolor="none")
    ax.axvline(0.05, color="#d62728", linewidth=1, linestyle="--")
    ax.set_title(region_label, fontsize=10)
    ax.set_xlabel("age p-value", fontsize=8)
    ax.set_ylabel("count", fontsize=8)

# hide unused
for j in range(len(REGION_ORDER), len(axes)):
    axes[j].axis("off")

fig.suptitle("Age p-value distribution for Immune/Inflammatory LR pairs\n"
             "(enrichment near 0 = signal; flat/uniform = no signal)",
             fontsize=12)
plt.tight_layout()
out = os.path.join(OUT_DIR, "immune_inflammatory_pval_hist.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
