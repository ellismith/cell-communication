#!/usr/bin/env python3
"""
Heatmap of Louvain combo direction agreement per functional category.

  rows = LR pairs
  columns = broad sender→receiver × region
  color = (n_pos - n_neg) / n_total  [-1 = all weakening, +1 = all strengthening]

One PNG per category saved to:
  results/within_region_analysis/hypergeometric_all_regions/category_plots/agreement/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob

# ── paths ──────────────────────────────────────────────────────────────────
CATEGORY_DIR = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/category_tables"
OUT_DIR      = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/category_plots/agreement"
os.makedirs(OUT_DIR, exist_ok=True)

REGION_ORDER  = ["acc", "cn", "dlpfc", "ec", "hip", "ipp", "lcb", "m1", "mb", "mdtn", "nac"]
REGION_LABELS = {
    "acc": "ACC", "cn": "CN", "dlpfc": "dlPFC", "ec": "EC", "hip": "HIP",
    "ipp": "IPP", "lcb": "lCB", "m1": "M1", "mb": "MB", "mdtn": "mdTN", "nac": "NAc",
}

MAX_FIG_W = 60
MAX_FIG_H = 40

# ── loop over category files ───────────────────────────────────────────────
category_files = sorted(glob.glob(os.path.join(CATEGORY_DIR, "category_*.csv")))
print(f"Found {len(category_files)} category files")

for fpath in category_files:
    cat_name = os.path.basename(fpath).replace("category_", "").replace(".csv", "").replace("_", " ")
    df = pd.read_csv(fpath)

    if len(df) == 0:
        continue

    # collapse Louvain to broad cell types
    df["sender_broad"]   = df["sender"].str.replace(r"_\d+$", "", regex=True)
    df["receiver_broad"] = df["receiver"].str.replace(r"_\d+$", "", regex=True)
    df["combo"] = df["sender_broad"] + " → " + df["receiver_broad"]

    # compute n_pos, n_neg, n_total per lr_pair × combo × region
    df["is_pos"] = (df["direction"] == "strengthening").astype(int)
    df["is_neg"] = (df["direction"] == "weakening").astype(int)

    agg = df.groupby(["lr_pair", "combo", "region"]).agg(
        n_pos=("is_pos", "sum"),
        n_neg=("is_neg", "sum"),
        n_total=("is_pos", "count")
    ).reset_index()

    agg["agreement"] = (agg["n_pos"] - agg["n_neg"]) / agg["n_total"]

    # build column label: combo | region
    regions = [r for r in REGION_ORDER if r in agg["region"].unique()]
    agg["col_label"] = agg["combo"] + " | " + agg["region"].map(
        lambda r: REGION_LABELS.get(r, r))

    # sort columns: alphabetically by combo, then by region order
    agg["region_order"] = agg["region"].map(
        {r: i for i, r in enumerate(REGION_ORDER)})
    agg = agg.sort_values(["combo", "region_order"])
    col_order = agg["col_label"].unique().tolist()

    # pivot
    pivot = agg.pivot(index="lr_pair", columns="col_label", values="agreement")
    pivot = pivot[col_order]
    pivot = pivot.sort_index()

    n_lr  = len(pivot)
    n_col = len(pivot.columns)

    fig_w = min(MAX_FIG_W, max(10, n_col * 0.25))
    fig_h = min(MAX_FIG_H, max(4,  n_lr  * 0.3))
    fontsize = max(4, min(7, int(7 * 20 / max(n_lr, n_col, 20))))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = plt.cm.RdBu_r.with_extremes(bad="lightgrey")

    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(n_col))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=fontsize)
    ax.set_yticks(range(n_lr))
    ax.set_yticklabels(pivot.index, fontsize=fontsize)
    ax.set_xlabel("Sender → Receiver | Region", fontsize=11)
    ax.set_ylabel("LR pair", fontsize=11)
    ax.set_title(
        f"{cat_name}\nDirection agreement: (n_strengthening - n_weakening) / n_total",
        fontsize=11
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.01)
    cbar.set_label("agreement score\n(+1=all strengthening, -1=all weakening)", fontsize=8)

    plt.tight_layout()

    fname = f"agreement_{cat_name.replace(' ', '_')}.png"
    fpath_out = os.path.join(OUT_DIR, fname)
    fig.savefig(fpath_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}  ({n_lr} LR pairs × {n_col} combos)")

print("Done.")
