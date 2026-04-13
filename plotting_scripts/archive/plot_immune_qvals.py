#!/usr/bin/env python3
"""
Two-panel dot plot of Immune/Inflammatory category enrichment:
  Top panel: raw p-values
  Bottom panel: FDR-corrected q-values
Both show strengthening and weakening, with dashed line at 0.05.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

CAT_ENRICH = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/hypergeometric_category_enrichment_all_regions.csv"
OUT_DIR    = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions"

REGION_ORDER = ["ACC", "CN", "dlPFC", "EC", "HIP", "IPP", "lCB", "M1", "MB", "mdTN", "NAc"]
REGION_LABELS = {
    "acc": "ACC", "cn": "CN", "dlpfc": "dlPFC", "ec": "EC", "hip": "HIP",
    "ipp": "IPP", "lcb": "lCB", "m1": "M1", "mb": "MB", "mdtn": "mdTN", "nac": "NAc",
}

cat_enrich = pd.read_csv(CAT_ENRICH)
imm = cat_enrich[cat_enrich["category"] == "Immune/Inflammatory"].copy()
imm["region_label"] = imm["region"].map(REGION_LABELS)
imm = imm[imm["region_label"].isin(REGION_ORDER)].copy()
imm["x"] = imm["region_label"].map({r: i for i, r in enumerate(REGION_ORDER)})

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

for ax, (p_str, p_wk, ylabel, title) in zip(axes, [
    ("p_pos_enrich", "p_neg_enrich", "p-value (uncorrected)",
     "Raw p-values"),
    ("q_pos_enrich", "q_neg_enrich", "q-value (FDR corrected)",
     "FDR-corrected q-values"),
]):
    ax.scatter(imm["x"] - 0.1, imm[p_str],
               color="#d62728", s=60, alpha=0.8, label="Strengthening", zorder=3)
    ax.scatter(imm["x"] + 0.1, imm[p_wk],
               color="#1f77b4", s=60, alpha=0.8, label="Weakening", zorder=3)
    ax.axhline(0.05, color="black", linewidth=1, linestyle="--",
               zorder=2, label="p/q = 0.05")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(-0.5, len(REGION_ORDER) - 0.5)

axes[1].set_xticks(range(len(REGION_ORDER)))
axes[1].set_xticklabels(REGION_ORDER, fontsize=10)
axes[1].set_xlabel("Region", fontsize=11)

fig.suptitle("Immune/Inflammatory signaling enrichment with age",
             fontsize=12, y=1.01)

plt.tight_layout()
out = os.path.join(OUT_DIR, "immune_inflammatory_pvals.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
