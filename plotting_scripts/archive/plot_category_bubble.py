#!/usr/bin/env python3
"""
Bubble plot for each significant category × region:
  x = LR pair
  y = sender→receiver (Louvain)
  size = abs(age_coef)
  color = direction (red=strengthening, blue=weakening)

One PNG per category × region saved to:
  results/within_region_analysis/hypergeometric_all_regions/category_plots/bubble/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import glob

# ── paths ──────────────────────────────────────────────────────────────────
CATEGORY_DIR = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/category_tables"
OUT_DIR      = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/category_plots/bubble"
os.makedirs(OUT_DIR, exist_ok=True)

REGION_LABELS = {
    "acc":   "ACC",  "cn":    "CN",   "dlpfc": "dlPFC",
    "ec":    "EC",   "hip":   "HIP",  "ipp":   "IPP",
    "lcb":   "lCB",  "m1":    "M1",   "mb":    "MB",
    "mdtn":  "mdTN", "nac":   "NAc",
}

# ── loop over category files ───────────────────────────────────────────────
category_files = sorted(glob.glob(os.path.join(CATEGORY_DIR, "category_*.csv")))
print(f"Found {len(category_files)} category files")

for fpath in category_files:
    cat_name = os.path.basename(fpath).replace("category_", "").replace(".csv", "").replace("_", " ")
    df = pd.read_csv(fpath)

    for region, rdf in df.groupby("region"):
        if len(rdf) == 0:
            continue

        rdf = rdf.copy()
        rdf["combo"] = rdf["sender"] + " → " + rdf["receiver"]

        # sort LR pairs and combos alphabetically
        lr_pairs = sorted(rdf["lr_pair"].unique())
        combos   = sorted(rdf["combo"].unique())

        lr_idx    = {lr: i for i, lr in enumerate(lr_pairs)}
        combo_idx = {c: i for i, c in enumerate(combos)}

        n_lr    = len(lr_pairs)
        n_combo = len(combos)

        fig_w = max(8, n_lr * 0.4)
        fig_h = max(4, n_combo * 0.35)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        for _, row in rdf.iterrows():
            x     = lr_idx[row["lr_pair"]]
            y     = combo_idx[row["combo"]]
            size  = abs(row["age_coef"]) * 3000
            color = "#d62728" if row["direction"] == "strengthening" else "#1f77b4"
            ax.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors="none")

        ax.set_xticks(range(n_lr))
        ax.set_xticklabels(lr_pairs, rotation=90, fontsize=7)
        ax.set_yticks(range(n_combo))
        ax.set_yticklabels(combos, fontsize=7)
        ax.set_xlabel("LR pair", fontsize=11)
        ax.set_ylabel("Sender → Receiver", fontsize=11)
        ax.set_title(f"{cat_name} — {REGION_LABELS.get(region, region)}", fontsize=12)

        # legend for color
        patches = [
            mpatches.Patch(color="#d62728", label="Strengthening"),
            mpatches.Patch(color="#1f77b4", label="Weakening"),
        ]
        # legend for size
        example_sizes = [0.005, 0.01, 0.02]
        size_handles = [
            plt.scatter([], [], s=s*3000, c="grey", alpha=0.7, edgecolors="none",
                        label=f"|β|={s}") for s in example_sizes
        ]
        ax.legend(handles=patches + size_handles, loc="upper right",
                  fontsize=8, framealpha=0.8)

        ax.set_xlim(-0.5, n_lr - 0.5)
        ax.set_ylim(-0.5, n_combo - 0.5)
        ax.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()

        region_label = REGION_LABELS.get(region, region)
        fname = f"bubble_{cat_name.replace(' ', '_')}_{region_label}.png"
        fpath_out = os.path.join(OUT_DIR, fname)
        fig.savefig(fpath_out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")

print("Done.")
