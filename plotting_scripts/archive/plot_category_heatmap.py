#!/usr/bin/env python3
"""
Heatmap for each significant category × region:
  rows = LR pairs
  columns = sender→receiver (Louvain)
  color = age_coef (diverging, red=strengthening, blue=weakening)

One PNG per category × region saved to:
  results/within_region_analysis/hypergeometric_all_regions/category_plots/heatmap/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob

# ── paths ──────────────────────────────────────────────────────────────────
CATEGORY_DIR = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/category_tables"
OUT_DIR      = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/category_plots/heatmap"
os.makedirs(OUT_DIR, exist_ok=True)

REGION_LABELS = {
    "acc":   "ACC",  "cn":    "CN",   "dlpfc": "dlPFC",
    "ec":    "EC",   "hip":   "HIP",  "ipp":   "IPP",
    "lcb":   "lCB",  "m1":    "M1",   "mb":    "MB",
    "mdtn":  "mdTN", "nac":   "NAc",
}

MAX_FIG_W = 40
MAX_FIG_H = 40

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

        # pivot: rows = lr_pair, cols = combo, values = age_coef
        pivot = rdf.pivot(index="lr_pair", columns="combo", values="age_coef")
        pivot = pivot.sort_index(axis=0).sort_index(axis=1)

        n_lr    = len(pivot)
        n_combo = len(pivot.columns)

        fig_w = min(MAX_FIG_W, max(6, n_combo * 0.4))
        fig_h = min(MAX_FIG_H, max(4, n_lr * 0.35))

        # scale fontsize down for large plots
        fontsize = max(4, min(7, int(7 * 20 / max(n_lr, n_combo, 20))))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        finite_vals = pivot.values[np.isfinite(pivot.values)]
        vmax = np.nanpercentile(np.abs(finite_vals), 95) if len(finite_vals) else 0.001
        vmax = max(vmax, 0.001)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cmap = plt.cm.RdBu_r.with_extremes(bad="lightgrey")

        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)

        ax.set_xticks(range(n_combo))
        ax.set_xticklabels(pivot.columns, rotation=90, fontsize=fontsize)
        ax.set_yticks(range(n_lr))
        ax.set_yticklabels(pivot.index, fontsize=fontsize)
        ax.set_xlabel("Sender → Receiver", fontsize=11)
        ax.set_ylabel("LR pair", fontsize=11)
        ax.set_title(f"{cat_name} — {REGION_LABELS.get(region, region)}", fontsize=12)

        cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("age_coef", fontsize=9)

        plt.tight_layout()

        region_label = REGION_LABELS.get(region, region)
        fname = f"heatmap_{cat_name.replace(' ', '_')}_{region_label}.png"
        fpath_out = os.path.join(OUT_DIR, fname)
        fig.savefig(fpath_out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}  ({n_lr} LR pairs × {n_combo} combos)")

print("Done.")
