#!/usr/bin/env python3
"""
For each significant category × region (from master_summary_table.csv),
plot a heatmap of age_coef at Louvain level.
Columns filtered to only Louvain combos where sender OR receiver
is in the hypergeometric-significant Louvain set.
Capped at 300 combos (most populated).
Both rows and columns clustered by correlation.

  rows = LR pairs (age-sig in either direction)
  cols = Louvain sender → Louvain receiver (filtered)
  color = age_coef (red=strengthening, blue=weakening)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = "/scratch/easmit31/cell_cell/results/within_region_analysis_corrected"
SUMMARY_FILE  = os.path.join(BASE_DIR, "hypergeometric_all_regions", "master_summary_table.csv")
OVERLAP_FILE  = os.path.join(BASE_DIR, "hypergeometric_all_regions", "louvain_overlap_check.csv")
CATEGORY_DIR  = os.path.join(BASE_DIR, "hypergeometric_all_regions", "category_tables")
OUT_DIR       = os.path.join(BASE_DIR, "hypergeometric_all_regions", "louvain_heatmaps")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_FIG_W  = 40
MAX_FIG_H  = 40
MAX_COMBOS = 300

REGION_LABELS_INV = {
    "ACC": "acc", "CN": "cn", "dlPFC": "dlpfc", "EC": "ec", "HIP": "hip",
    "IPP": "ipp", "lCB": "lcb", "M1": "m1", "MB": "mb", "mdTN": "mdtn", "NAc": "nac",
}

def reorder_by_correlation(pivot, axis=1):
    """Reorder rows (axis=0) or columns (axis=1) by greedy nearest-neighbor correlation."""
    if axis == 1:
        data = pivot.fillna(0)
    else:
        data = pivot.fillna(0).T
    if data.shape[1] <= 2:
        return pivot
    corr = data.corr()
    remaining = list(corr.columns)
    ordered = [remaining.pop(0)]
    while remaining:
        last = ordered[-1]
        sims = corr[last][remaining]
        next_col = sims.idxmax()
        ordered.append(next_col)
        remaining.remove(next_col)
    if axis == 1:
        return pivot[ordered]
    else:
        return pivot.loc[ordered]

# ── load ───────────────────────────────────────────────────────────────────
summary = pd.read_csv(SUMMARY_FILE)
overlap = pd.read_csv(OVERLAP_FILE)

sig_louvains = {}
for _, row in overlap.iterrows():
    key  = (row["region"], row["cell_type"], row["role"], row["direction"])
    lous = set(row["overlap"].split(", ")) \
           if pd.notna(row["overlap"]) and row["overlap"] else set()
    sig_louvains[key] = lous

cat_files = sorted(glob.glob(os.path.join(CATEGORY_DIR, "category_*.csv")))
cat_dfs = {}
for f in cat_files:
    cat_name = os.path.basename(f).replace("category_", "").replace(".csv", "").replace("_", " ")
    df = pd.read_csv(f)
    cat_dfs[cat_name] = df

sig_combos = summary[["region", "category"]].drop_duplicates()
print(f"Plotting {len(sig_combos)} category × region combos")

# ── main loop ──────────────────────────────────────────────────────────────
for _, crow in sig_combos.iterrows():
    region_label  = crow["region"]
    category      = crow["category"]
    region_key    = REGION_LABELS_INV.get(region_label, region_label.lower())

    if category not in cat_dfs:
        continue

    cat_df        = cat_dfs[category]
    cat_df_region = cat_df[cat_df["region"] == region_key].copy()

    if len(cat_df_region) == 0:
        continue

    cat_sig    = summary[(summary["region"] == region_label) &
                         (summary["category"] == category)]
    directions = cat_sig["direction"].unique()

    sig_senders   = set()
    sig_receivers = set()
    for direction in directions:
        for broad_ct in cat_df_region["sender"].str.replace(r"_\d+$","",regex=True).unique():
            k = (region_key, broad_ct, "sender", direction)
            sig_senders |= sig_louvains.get(k, set())
        for broad_ct in cat_df_region["receiver"].str.replace(r"_\d+$","",regex=True).unique():
            k = (region_key, broad_ct, "receiver", direction)
            sig_receivers |= sig_louvains.get(k, set())

    mask = (
        cat_df_region["sender"].isin(sig_senders) |
        cat_df_region["receiver"].isin(sig_receivers)
    )
    sub = cat_df_region[mask].copy()

    if len(sub) == 0:
        print(f"  Skipping {category} — {region_label}: no sig Louvain combos after filter")
        continue

    sub["combo"] = sub["sender"] + " → " + sub["receiver"]

    if sub["combo"].nunique() > MAX_COMBOS:
        top_combos = (sub.groupby("combo").size()
                      .sort_values(ascending=False)
                      .head(MAX_COMBOS).index)
        sub = sub[sub["combo"].isin(top_combos)]
        print(f"  Capped to top {MAX_COMBOS} combos for {category} — {region_label}")

    pivot = sub.pivot_table(
        index="lr_pair", columns="combo",
        values="age_coef", aggfunc="mean"
    )

    # cluster both rows and columns by correlation
    if pivot.shape[1] > 2:
        pivot = reorder_by_correlation(pivot, axis=1)  # columns
    if pivot.shape[0] > 2:
        pivot = reorder_by_correlation(pivot, axis=0)  # rows

    n_lr    = len(pivot)
    n_combo = len(pivot.columns)

    if n_lr == 0 or n_combo == 0:
        continue

    fig_w    = min(MAX_FIG_W, max(6, n_combo * 0.25))
    fig_h    = min(MAX_FIG_H, max(4, n_lr    * 0.3))
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
    ax.set_xlabel("Louvain Sender → Receiver", fontsize=10)
    ax.set_ylabel("LR pair", fontsize=10)
    ax.set_title(
        f"{category} — {region_label}\n"
        f"({n_lr} LR pairs × {n_combo} Louvain combos, "
        f"red=strengthening, blue=weakening)",
        fontsize=10
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.01)
    cbar.set_label("age_coef", fontsize=8)

    plt.tight_layout()

    fname = f"louvain_{category.replace(' ', '_')}_{region_label}.png"
    fpath = os.path.join(OUT_DIR, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}  ({n_lr} × {n_combo})")

print("Done.")
