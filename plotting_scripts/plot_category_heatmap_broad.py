#!/usr/bin/env python3
"""
Heatmap for each significant category × region, with Louvain clusters
collapsed to broad cell types (age_coef averaged across Louvains).

  rows = LR pairs
  columns = broad sender→receiver
  color = mean age_coef across Louvain clusters (diverging, red=strengthening, blue=weakening)

Usage:
  python plot_category_heatmap_broad.py
  python plot_category_heatmap_broad.py --region hip
  python plot_category_heatmap_broad.py --category "Axon guidance"
  python plot_category_heatmap_broad.py --region hip --category "Axon guidance"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import os
import glob

# ── args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--region",   type=str, default=None)
parser.add_argument("--category", type=str, default=None)
args = parser.parse_args()

# ── paths ──────────────────────────────────────────────────────────────────
CATEGORY_DIR = "/scratch/easmit31/cell_cell/results/within_region_analysis_corrected/hypergeometric_all_regions/category_tables"
OUT_DIR      = "/scratch/easmit31/cell_cell/results/within_region_analysis_corrected/hypergeometric_all_regions/category_plots/heatmap_broad"
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

if args.category:
    cat_fname = "category_" + args.category.replace(" ", "_").replace("/", "_") + ".csv"
    category_files = [f for f in category_files if os.path.basename(f) == cat_fname]
    if not category_files:
        print(f"ERROR: no category file found matching '{args.category}'")
        exit(1)

print(f"Found {len(category_files)} category file(s) to process")

for fpath in category_files:
    cat_name = os.path.basename(fpath).replace("category_", "").replace(".csv", "").replace("_", " ")
    df = pd.read_csv(fpath)

    if args.region:
        df = df[df["region"] == args.region.lower()]
        if len(df) == 0:
            print(f"  WARNING: no data for region '{args.region}' in {cat_name}")
            continue

    for region, rdf in df.groupby("region"):
        rdf = rdf.copy()
        region_label = REGION_LABELS.get(region, region)

        # collapse Louvain to broad cell types
        rdf["sender_broad"]   = rdf["sender"].str.replace(r"_\d+$", "", regex=True)
        rdf["receiver_broad"] = rdf["receiver"].str.replace(r"_\d+$", "", regex=True)
        rdf["combo"] = rdf["sender_broad"] + " → " + rdf["receiver_broad"]

        # average age_coef across Louvain clusters
        pivot = rdf.groupby(["lr_pair", "combo"])["age_coef"].mean().unstack()
        pivot = pivot.sort_index(axis=0).sort_index(axis=1)

        n_lr    = len(pivot)
        n_combo = len(pivot.columns)

        fig_w = min(MAX_FIG_W, max(6, n_combo * 0.6))
        fig_h = min(MAX_FIG_H, max(4, n_lr * 0.4))
        fontsize = max(4, min(9, int(9 * 20 / max(n_lr, n_combo, 20))))

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
        ax.set_title(f"{cat_name} — {region_label} (broad cell types)", fontsize=12)

        cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("mean age_coef", fontsize=9)

        plt.tight_layout()

        fname = f"heatmap_broad_{cat_name.replace(' ', '_')}_{region_label}.png"
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}  ({n_lr} LR pairs × {n_combo} combos)")

print("Done.")
