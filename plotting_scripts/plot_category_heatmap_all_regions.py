#!/usr/bin/env python3
"""
One combined heatmap per functional category.
All regions and both directions (strengthening/weakening) in one plot.
Rows = LR pairs (aligned across regions).
Columns grouped by region, within each region clustered by correlation.
Color = mean age_coef (red=strengthening, blue=weakening), grey=not significant.
Only significant category × region × direction combos shown (q<0.05).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import os
import glob

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = "/scratch/easmit31/cell_cell/results/within_region_analysis_corrected"
CAT_ENRICH   = os.path.join(BASE_DIR, "hypergeometric_all_regions",
                            "hypergeometric_category_enrichment_all_regions.csv")
CATEGORY_DIR = os.path.join(BASE_DIR, "hypergeometric_all_regions", "category_tables")
OUT_DIR      = os.path.join(BASE_DIR, "hypergeometric_all_regions",
                            "heatmaps_all_regions_combined")
os.makedirs(OUT_DIR, exist_ok=True)

Q_THRESH  = 0.05
MAX_LR    = 40
MAX_FIG_W = 60
MAX_FIG_H = 30

REGION_ORDER = ["ACC", "NAc", "CN", "dlPFC", "EC", "HIP", "IPP", "M1", "MB", "mdTN", "lCB"]
REGION_LABELS_INV = {
    "ACC": "acc", "CN": "cn", "dlPFC": "dlpfc", "EC": "ec", "HIP": "hip",
    "IPP": "ipp", "lCB": "lcb", "M1": "m1", "MB": "mb", "mdTN": "mdtn", "NAc": "nac",
}

DIRECTION_COLORS = {"strengthening": "#d62728", "weakening": "#1f77b4"}

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

# ── get significant combos ─────────────────────────────────────────────────
sig_rows = []
for _, row in cat_enrich.iterrows():
    for direction, qcol, fecol in [
        ("strengthening", "q_pos_enrich", "fold_enrichment_pos"),
        ("weakening",     "q_neg_enrich", "fold_enrichment_neg"),
    ]:
        if row[qcol] < Q_THRESH:
            sig_rows.append({
                "region":    row["region"],
                "category":  row["category"],
                "direction": direction,
            })

sig = pd.DataFrame(sig_rows)
categories = sig["category"].unique()
print(f"Found {len(sig)} significant combos across {len(categories)} categories")

# ── correlation-based column reordering ───────────────────────────────────
def reorder_by_correlation(pivot):
    if pivot.shape[1] <= 2:
        return pivot
    filled    = pivot.fillna(0)
    corr      = filled.corr()
    remaining = list(corr.columns)
    ordered   = [remaining.pop(0)]
    while remaining:
        last     = ordered[-1]
        sims     = corr[last][remaining]
        next_col = sims.idxmax()
        ordered.append(next_col)
        remaining.remove(next_col)
    return pivot[ordered]

# ── main loop ─────────────────────────────────────────────────────────────
for category in sorted(categories):
    if category not in cat_dfs:
        continue

    cat_df   = cat_dfs[category]
    sig_cat  = sig[sig["category"] == category]

    # collect all LR pairs that appear in any sig region/direction
    all_lrs = set()
    for _, srow in sig_cat.iterrows():
        region_key = REGION_LABELS_INV.get(srow["region"], srow["region"].lower())
        sub = cat_df[
            (cat_df["region"] == region_key) &
            (cat_df["direction"] == srow["direction"])
        ]
        all_lrs |= set(sub["lr_pair"].unique())

    # cap at MAX_LR by total frequency across all regions
    all_sub = cat_df[cat_df["lr_pair"].isin(all_lrs)]
    top_lrs = (all_sub.groupby("lr_pair").size()
               .sort_values(ascending=False)
               .head(MAX_LR).index.tolist())

    if not top_lrs:
        continue

    # build one pivot per region × direction, then concatenate columns
    region_pivots  = {}  # region_label -> pivot
    region_dirs    = {}  # region_label -> list of directions present
    col_direction  = {}  # col_name -> direction (for coloring)

    for region_label in REGION_ORDER:
        region_key = REGION_LABELS_INV.get(region_label, region_label.lower())
        sig_here   = sig_cat[sig_cat["region"] == region_label]
        if len(sig_here) == 0:
            continue

        region_cols = {}
        for _, srow in sig_here.iterrows():
            direction = srow["direction"]
            sub = cat_df[
                (cat_df["region"] == region_key) &
                (cat_df["direction"] == direction) &
                (cat_df["lr_pair"].isin(top_lrs))
            ].copy()
            if len(sub) == 0:
                continue

            sub["combo"] = sub["sender_broad"] + "→" + sub["receiver_broad"]
            pivot_dir = sub.groupby(["lr_pair","combo"])["age_coef"].mean().unstack()
            pivot_dir = pivot_dir.reindex(index=top_lrs)

            # rename cols to include direction tag for uniqueness
            rename = {c: f"{c} [{direction[:3]}]" for c in pivot_dir.columns}
            pivot_dir = pivot_dir.rename(columns=rename)
            for c in pivot_dir.columns:
                col_direction[c] = direction

            region_cols[direction] = pivot_dir

        if not region_cols:
            continue

        # combine directions for this region, cluster cols by correlation
        region_pivot = pd.concat(list(region_cols.values()), axis=1)
        region_pivot = reorder_by_correlation(region_pivot)
        region_pivots[region_label] = region_pivot

    if not region_pivots:
        continue

    # concatenate all regions
    full_pivot = pd.concat(list(region_pivots.values()), axis=1)
    full_pivot = full_pivot.reindex(index=top_lrs)

    n_lr    = len(full_pivot)
    n_combo = len(full_pivot.columns)

    fig_w = min(MAX_FIG_W, max(8, n_combo * 0.35))
    fig_h = min(MAX_FIG_H, max(4, n_lr    * 0.32))
    fs    = max(4, min(7, int(7 * 20 / max(n_lr, n_combo, 20))))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    finite_vals = full_pivot.values[np.isfinite(full_pivot.values)]
    vmax = np.nanpercentile(np.abs(finite_vals), 95) if len(finite_vals) else 0.001
    vmax = max(vmax, 0.001)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r.with_extremes(bad="#eeeeee")

    im = ax.imshow(full_pivot.values, aspect="auto", cmap=cmap, norm=norm)

    # x-tick labels colored by direction
    ax.set_xticks(range(n_combo))
    ax.set_xticklabels(full_pivot.columns, rotation=90, fontsize=fs)
    for tick, col in zip(ax.get_xticklabels(), full_pivot.columns):
        d = col_direction.get(col, "strengthening")
        tick.set_color(DIRECTION_COLORS[d])

    ax.set_yticks(range(n_lr))
    ax.set_yticklabels(full_pivot.index, fontsize=fs)

    # draw vertical lines between regions
    x = 0
    for region_label in REGION_ORDER:
        if region_label not in region_pivots:
            continue
        w = len(region_pivots[region_label].columns)
        if x > 0:
            ax.axvline(x - 0.5, color="black", lw=1.5)
        ax.text(x + w/2 - 0.5, -1.5, region_label,
                ha="center", va="top", fontsize=8, fontweight="bold",
                transform=ax.get_xaxis_transform())
        x += w

    ax.set_xlabel("Broad Sender → Receiver  (grouped by region)", fontsize=9, labelpad=20)
    ax.set_ylabel("LR pair", fontsize=9)
    ax.set_title(f"{category} — all regions\n"
                 f"(top {min(MAX_LR, n_lr)} LR pairs, columns clustered by correlation within region)",
                 fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.01)
    cbar.set_label("mean age_coef", fontsize=8)

    patches = [mpatches.Patch(color=c, label=d)
               for d, c in DIRECTION_COLORS.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.8)

    plt.tight_layout()

    fname = f"combined_{category.replace(' ','_')}_all_regions.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}  ({n_lr} LR pairs × {n_combo} cols)")

print("Done.")
