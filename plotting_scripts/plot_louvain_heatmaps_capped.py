#!/usr/bin/env python3
"""
Same as plot_louvain_heatmaps.py but capped at 50 rows (top LR pairs by
absolute age_coef) and 50 columns (most populated combos).
Both axes clustered by correlation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob

BASE_DIR      = "/scratch/easmit31/cell_cell/results/within_region_analysis_corrected"
SUMMARY_FILE  = os.path.join(BASE_DIR, "hypergeometric_all_regions", "master_summary_table.csv")
OVERLAP_FILE  = os.path.join(BASE_DIR, "hypergeometric_all_regions", "louvain_overlap_check.csv")
CATEGORY_DIR  = os.path.join(BASE_DIR, "hypergeometric_all_regions", "category_tables")
OUT_DIR       = os.path.join(BASE_DIR, "hypergeometric_all_regions", "louvain_heatmaps_capped")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_FIG_W  = 40
MAX_FIG_H  = 40
MAX_COMBOS = 50
MAX_LR     = 50

REGION_LABELS_INV = {
    "ACC": "acc", "CN": "cn", "dlPFC": "dlpfc", "EC": "ec", "HIP": "hip",
    "IPP": "ipp", "lCB": "lcb", "M1": "m1", "MB": "mb", "mdTN": "mdtn", "NAc": "nac",
}

def reorder_by_correlation(pivot, axis=1):
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
        sims = corr[last][remaining].dropna()
        if len(sims) == 0:
            ordered.extend(remaining)
            break
        best = sims.idxmax()
        ordered.append(best)
        remaining.remove(best)
    if axis == 1:
        return pivot[ordered]
    else:
        return pivot.loc[ordered]

summary = pd.read_csv(SUMMARY_FILE)
overlap  = pd.read_csv(OVERLAP_FILE)

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
    cat_dfs[cat_name] = pd.read_csv(f)

sig_combos = summary[["region", "category"]].drop_duplicates()
print(f"Plotting {len(sig_combos)} category × region combos")

for _, crow in sig_combos.iterrows():
    region_label = crow["region"]
    category     = crow["category"]
    region_key   = REGION_LABELS_INV.get(region_label, region_label.lower())

    if category not in cat_dfs:
        continue

    cat_df_region = cat_dfs[category][cat_dfs[category]["region"] == region_key].copy()
    if len(cat_df_region) == 0:
        continue

    cat_sig    = summary[(summary["region"] == region_label) & (summary["category"] == category)]
    directions = cat_sig["direction"].unique()

    sig_senders, sig_receivers = set(), set()
    for direction in directions:
        for broad_ct in cat_df_region["sender"].str.replace(r"_\d+$","",regex=True).unique():
            sig_senders |= sig_louvains.get((region_key, broad_ct, "sender", direction), set())
        for broad_ct in cat_df_region["receiver"].str.replace(r"_\d+$","",regex=True).unique():
            sig_receivers |= sig_louvains.get((region_key, broad_ct, "receiver", direction), set())

    sub = cat_df_region[
        cat_df_region["sender"].isin(sig_senders) |
        cat_df_region["receiver"].isin(sig_receivers)
    ].copy()

    if len(sub) == 0:
        continue

    sub["combo"] = sub["sender"] + " → " + sub["receiver"]

    if sub["combo"].nunique() > MAX_COMBOS:
        top_combos = (sub.groupby("combo").size()
                      .sort_values(ascending=False)
                      .head(MAX_COMBOS).index)
        sub = sub[sub["combo"].isin(top_combos)]

    pivot = sub.pivot_table(
        index="lr_pair", columns="combo",
        values="age_coef", aggfunc="mean"
    )

    if len(pivot) > MAX_LR:
        top_lrs = pivot.abs().mean(axis=1).nlargest(MAX_LR).index
        pivot = pivot.loc[top_lrs]

    if pivot.shape[1] > 2:
        pivot = reorder_by_correlation(pivot, axis=1)
    if pivot.shape[0] > 2:
        pivot = reorder_by_correlation(pivot, axis=0)

    n_lr, n_combo = len(pivot), len(pivot.columns)
    if n_lr == 0 or n_combo == 0:
        continue

    fig_w    = min(MAX_FIG_W, max(6, n_combo * 0.25))
    fig_h    = min(MAX_FIG_H, max(4, n_lr * 0.3))
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
        f"(top {n_lr} LR pairs × top {n_combo} Louvain combos, "
        f"red=strengthening, blue=weakening)",
        fontsize=10
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.01)
    cbar.set_label("age_coef", fontsize=8)

    plt.tight_layout()
    fname = f"louvain_{category.replace(' ', '_')}_{region_label}.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}  ({n_lr} × {n_combo})")

print("Done.")
