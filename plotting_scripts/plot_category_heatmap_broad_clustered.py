#!/usr/bin/env python3
"""
Broad cell type heatmaps per significant category × region × direction.
Columns (broad sender→receiver) reordered by correlation of LR pair profiles.
LR pairs capped at top 30 by frequency.
Strengthening and weakening shown as separate plots.

Only plots significant category × region × direction combos (q<0.05
from hypergeometric_category_enrichment_all_regions.csv).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = "/scratch/easmit31/cell_cell/results/within_region_analysis_corrected"
CAT_ENRICH    = os.path.join(BASE_DIR, "hypergeometric_category",
                             "hypergeometric_category_enrichment_all_regions.csv")
CATEGORY_DIR  = os.path.join(BASE_DIR, "hypergeometric_category", "category_tables")
OUT_DIR       = os.path.join(BASE_DIR, "hypergeometric_category", "heatmaps_broad_clustered")
os.makedirs(OUT_DIR, exist_ok=True)

Q_THRESH  = 0.05
MAX_LR    = 30
MAX_FIG_W = 40
MAX_FIG_H = 20

REGION_LABELS = {
    "acc": "ACC", "cn": "CN", "dlpfc": "dlPFC", "ec": "EC", "hip": "HIP",
    "ipp": "IPP", "lcb": "lCB", "m1": "M1", "mb": "MB", "mdtn": "mdTN", "nac": "NAc",
}

# ── load ───────────────────────────────────────────────────────────────────
cat_enrich = pd.read_csv(CAT_ENRICH)

cat_files = sorted(glob.glob(os.path.join(CATEGORY_DIR, "category_*.csv")))
cat_dfs = {}
for f in cat_files:
    cat_name = os.path.basename(f).replace("category_","").replace(".csv","").replace("_"," ")
    df = pd.read_csv(f)
    df["sender_broad"]   = df["sender"].str.replace(r"_\d+$","",regex=True)
    df["receiver_broad"] = df["receiver"].str.replace(r"_\d+$","",regex=True)
    cat_dfs[cat_name] = df

print(f"Loaded {len(cat_dfs)} category tables")

# ── get significant combos ─────────────────────────────────────────────────
sig_rows = []
for _, row in cat_enrich.iterrows():
    for direction, qcol, fecol in [
        ("strengthening", "q_pos_enrich", "fold_enrichment_pos"),
        ("weakening",     "q_neg_enrich", "fold_enrichment_neg"),
    ]:
        if row[qcol] < Q_THRESH:
            sig_rows.append({
                "region":        row["region"],
                "category":      row["category"],
                "direction":     direction,
                "fold_enrichment": row[fecol],
                "q_val":         row[qcol],
            })

sig = pd.DataFrame(sig_rows)
print(f"Found {len(sig)} significant category × region × direction combos")

# ── correlation-based column reordering ───────────────────────────────────
def reorder_by_correlation(pivot):
    if pivot.shape[1] <= 2:
        return pivot
    filled = pivot.fillna(0)
    corr   = filled.corr()
    remaining = list(corr.columns)
    ordered   = [remaining.pop(0)]
    while remaining:
        last = ordered[-1]
        sims = corr[last][remaining]
        next_col = sims.idxmax()
        ordered.append(next_col)
        remaining.remove(next_col)
    return pivot[ordered]

# ── main loop ──────────────────────────────────────────────────────────────
for _, srow in sig.iterrows():
    region_key   = srow["region"]
    category     = srow["category"]
    direction    = srow["direction"]
    fe           = srow["fold_enrichment"]
    q_val        = srow["q_val"]
    region_label = REGION_LABELS.get(region_key, region_key.upper())

    # match category table filename
    cat_key = category.replace(' ','_').replace('/','_')
    matched = [k for k in cat_dfs if k.replace(' ','_').replace('/','_') == cat_key]
    if not matched:
        print(f"  No table for: {category}")
        continue

    cat_df = cat_dfs[matched[0]]
    sub = cat_df[
        (cat_df["region"] == region_key) &
        (cat_df["direction"] == direction)
    ].copy()

    if len(sub) == 0:
        continue

    sub["combo"] = sub["sender_broad"] + " → " + sub["receiver_broad"]

    top_lrs = (sub.groupby("lr_pair").size()
               .sort_values(ascending=False)
               .head(MAX_LR).index.tolist())
    sub = sub[sub["lr_pair"].isin(top_lrs)]

    pivot = sub.groupby(["lr_pair","combo"])["age_coef"].mean().unstack()
    pivot = reorder_by_correlation(pivot.T).T
    if pivot.shape[0] == 0 or pivot.shape[1] == 0:
        continue
    pivot = reorder_by_correlation(pivot)

    n_lr    = len(pivot)
    n_combo = len(pivot.columns)

    fig_w    = min(MAX_FIG_W, max(6, n_combo * 0.4))
    fig_h    = min(MAX_FIG_H, max(4, n_lr * 0.35))
    fontsize = max(5, min(8, int(8 * 20 / max(n_lr, n_combo, 20))))

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
    ax.set_xlabel("Broad Sender → Receiver", fontsize=10)
    ax.set_ylabel("LR pair", fontsize=10)
    ax.set_title(
        f"{category} — {region_label} — {direction}\n"
        f"FE={fe:.2f}  q={q_val:.2e}  "
        f"(top {min(MAX_LR, n_lr)} LR pairs, columns ordered by correlation)",
        fontsize=9
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.01)
    cbar.set_label("mean age_coef", fontsize=8)

    plt.tight_layout()

    fname = f"heatmap_{cat_key}_{region_label}_{direction}.png"
    fpath = os.path.join(OUT_DIR, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}  ({n_lr} LR pairs × {n_combo} combos)")

print("Done.")
