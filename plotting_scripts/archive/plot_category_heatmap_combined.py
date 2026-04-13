#!/usr/bin/env python3
"""
For categories significant in BOTH strengthening and weakening in a region,
ONE heatmap:
  - rows = top LR pairs (by frequency), ordered by mean age_coef
  - cols = all unique broad sender→receiver combos, corr-clustered, NO duplicates
  - color = mean age_coef (red=strengthening, blue=weakening)
  - purple+bold column label if combo has both str and wk LR pairs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob

BASE_DIR     = "/scratch/easmit31/cell_cell/results/within_region_analysis"
CAT_ENRICH   = os.path.join(BASE_DIR, "hypergeometric_all_regions",
                            "hypergeometric_category_enrichment_all_regions.csv")
CATEGORY_DIR = os.path.join(BASE_DIR, "hypergeometric_all_regions", "category_tables")
OUT_DIR      = os.path.join(BASE_DIR, "hypergeometric_all_regions", "heatmaps_broad_combined")
os.makedirs(OUT_DIR, exist_ok=True)

Q_THRESH  = 0.05
MAX_LR    = 30
MAX_FIG_W = 40
MAX_FIG_H = 20

REGION_LABELS = {
    "acc": "ACC", "cn": "CN", "dlpfc": "dlPFC", "ec": "EC", "hip": "HIP",
    "ipp": "IPP", "lcb": "lCB", "m1": "M1", "mb": "MB", "mdtn": "mdTN", "nac": "NAc",
}

cat_enrich = pd.read_csv(CAT_ENRICH)
cat_files  = sorted(glob.glob(os.path.join(CATEGORY_DIR, "category_*.csv")))
cat_dfs = {}
for f in cat_files:
    cat_name = os.path.basename(f).replace("category_","").replace(".csv","").replace("_"," ")
    df = pd.read_csv(f)
    df["sender_broad"]   = df["sender"].str.replace(r"_\d+$","",regex=True)
    df["receiver_broad"] = df["receiver"].str.replace(r"_\d+$","",regex=True)
    df["combo"]          = df["sender_broad"] + " → " + df["receiver_broad"]
    cat_dfs[cat_name]    = df

both_sig = cat_enrich[
    (cat_enrich["q_pos_enrich"] < Q_THRESH) &
    (cat_enrich["q_neg_enrich"] < Q_THRESH)
]

def reorder_cols(pivot):
    if pivot.shape[1] <= 2:
        return pivot
    filled    = pivot.fillna(0)
    corr      = filled.corr()
    remaining = list(corr.columns)
    ordered   = [remaining.pop(0)]
    while remaining:
        sims     = corr[ordered[-1]][remaining]
        next_col = sims.idxmax()
        ordered.append(next_col)
        remaining.remove(next_col)
    return pivot[ordered]

for _, srow in both_sig.iterrows():
    region_key   = srow["region"]
    category     = srow["category"]
    region_label = REGION_LABELS.get(region_key, region_key.upper())

    if category not in cat_dfs:
        continue

    sub = cat_dfs[category][cat_dfs[category]["region"] == region_key].copy()
    if len(sub) == 0:
        continue

    # top LR pairs by frequency across both directions
    top_lrs = (sub.groupby("lr_pair").size()
               .sort_values(ascending=False).head(MAX_LR).index.tolist())
    sub     = sub[sub["lr_pair"].isin(top_lrs)]

    # single pivot — no direction split, no duplicate columns
    pivot = sub.groupby(["lr_pair","combo"])["age_coef"].mean().unstack()

    # order rows by mean age_coef
    row_means   = sub.groupby("lr_pair")["age_coef"].mean()
    ordered_lrs = row_means.reindex(top_lrs).fillna(0).sort_values(ascending=False).index
    pivot       = pivot.reindex(ordered_lrs)

    # reorder cols by correlation
    pivot = reorder_cols(pivot)

    # find combos that have both str and wk LR pairs
    str_combos = set(sub[sub["direction"]=="strengthening"]["combo"].unique())
    wk_combos  = set(sub[sub["direction"]=="weakening"]["combo"].unique())
    both_combos = str_combos & wk_combos

    n_lr    = len(pivot)
    n_combo = len(pivot.columns)
    if n_lr == 0 or n_combo == 0:
        continue

    fig_w    = min(MAX_FIG_W, max(8, n_combo * 0.35))
    fig_h    = min(MAX_FIG_H, max(4, n_lr    * 0.35))
    fontsize = max(5, min(8, int(8 * 20 / max(n_lr, n_combo, 20))))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    finite_vals = pivot.values[np.isfinite(pivot.values)]
    vmax = np.nanpercentile(np.abs(finite_vals), 95) if len(finite_vals) else 0.001
    vmax = max(vmax, 0.001)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r.with_extremes(bad="lightgrey")

    ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(n_combo))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=fontsize)
    for tick, col in zip(ax.get_xticklabels(), pivot.columns):
        if col in both_combos:
            tick.set_color("#7b2d8b")
            tick.set_fontweight("bold")

    ax.set_yticks(range(n_lr))
    ax.set_yticklabels(pivot.index, fontsize=fontsize)
    ax.set_xlabel("Broad Sender → Receiver", fontsize=10)
    ax.set_ylabel("LR pair", fontsize=10)
    ax.set_title(
        f"{category} — {region_label}\n"
        f"Strengthening FE={srow['fold_enrichment_pos']:.2f} q={srow['q_pos_enrich']:.2e}  |  "
        f"Weakening FE={srow['fold_enrichment_neg']:.2f} q={srow['q_neg_enrich']:.2e}\n"
        f"Purple = combo has both strengthening and weakening LR pairs",
        fontsize=9
    )

    fig.colorbar(ax.images[0], ax=ax, shrink=0.4, pad=0.01,
                 label="mean age_coef")

    plt.tight_layout()
    fname = f"combined_{category.replace(' ','_')}_{region_label}.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}  ({n_lr} LR × {n_combo} combos)")

print("Done.")
