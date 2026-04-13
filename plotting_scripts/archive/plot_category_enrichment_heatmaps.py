#!/usr/bin/env python3
"""
Plot 2 heatmaps of functional category hypergeometric enrichment results:
  1. Strengthening (pos beta)
  2. Weakening (neg beta)

Each cell shows log2 (default) or log10 fold enrichment.
Positive values = enriched, negative values = depleted.
--sig_only: only show significant cells (q<0.05), others white
--clip: clip fold enrichment at 1 (log=0), only show enrichment

Usage:
  python plot_category_enrichment_heatmaps.py
  python plot_category_enrichment_heatmaps.py --log log10
  python plot_category_enrichment_heatmaps.py --clip
  python plot_category_enrichment_heatmaps.py --sig_only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import os

# ── args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--log", choices=["log2", "log10"], default="log2",
                    help="Log transform to apply to fold enrichment (default: log2)")
parser.add_argument("--clip", action="store_true",
                    help="Clip fold enrichment at 1 (log=0), only show enrichment")
parser.add_argument("--sig_only", action="store_true",
                    help="Only show significant cells (q<0.05); others white")
args = parser.parse_args()

log_fn    = np.log2 if args.log == "log2" else np.log10
log_label = f"{args.log}(fold enrichment)"
clip_tag  = "_clipped" if args.clip     else ""
sig_tag   = "_sigonly" if args.sig_only else ""

# ── paths ──────────────────────────────────────────────────────────────────
IN_PATH = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/hypergeometric_category_enrichment_all_regions.csv"
OUT_DIR = os.path.dirname(IN_PATH)

Q_THRESH = 0.05
REGION_ORDER  = ["acc", "cn", "dlpfc", "ec", "hip", "ipp", "lcb", "m1", "mb", "mdtn", "nac"]
REGION_LABELS = {
    "acc":   "ACC",
    "cn":    "CN",
    "dlpfc": "dlPFC",
    "ec":    "EC",
    "hip":   "HIP",
    "ipp":   "IPP",
    "lcb":   "lCB",
    "m1":    "M1",
    "mb":    "MB",
    "mdtn":  "mdTN",
    "nac":   "NAc",
}

# ── load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(IN_PATH)
print(f"Loaded {len(df)} rows")

# ── plotting function ──────────────────────────────────────────────────────
def make_heatmap(df, direction, log_fn, log_label, out_dir, log_name,
                 clip, clip_tag, sig_only, sig_tag):
    if direction == "strengthening":
        fe_col    = "fold_enrichment_pos"
        q_enrich  = "q_pos_enrich"
        q_deplete = "q_pos_deplete"
    else:
        fe_col    = "fold_enrichment_neg"
        q_enrich  = "q_neg_enrich"
        q_deplete = "q_neg_deplete"

    fe_pivot  = df.pivot(index="category", columns="region", values=fe_col)
    qe_pivot  = df.pivot(index="category", columns="region", values=q_enrich)
    qd_pivot  = df.pivot(index="category", columns="region", values=q_deplete)

    regions = [r for r in REGION_ORDER if r in fe_pivot.columns]
    fe_pivot  = fe_pivot[regions]
    qe_pivot  = qe_pivot[regions]
    qd_pivot  = qd_pivot[regions]

    # alphabetical category order
    categories = sorted(fe_pivot.index.tolist())
    fe_pivot  = fe_pivot.loc[categories]
    qe_pivot  = qe_pivot.loc[categories]
    qd_pivot  = qd_pivot.loc[categories]

    # mask non-significant cells if sig_only
    # use enrichment q for FE>=1, depletion q for FE<1
    if sig_only:
        sig_mask = np.where(
            fe_pivot.values >= 1,
            qe_pivot.values < Q_THRESH,
            qd_pivot.values < Q_THRESH,
        )
        fe_pivot = fe_pivot.where(sig_mask, other=np.nan)

    # clip at 1 if requested
    fe_plot = fe_pivot.clip(lower=1) if clip else fe_pivot.clip(lower=1e-6)
    log_fe  = log_fn(fe_plot.clip(lower=1e-6))

    # colormap and norm
    finite_vals = log_fe.values[np.isfinite(log_fe.values)]
    if clip:
        cmap = plt.cm.Reds.with_extremes(bad="white")
        vmax = np.nanpercentile(finite_vals, 95) if len(finite_vals) else 1
        vmax = max(vmax, 0.5)
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
    else:
        cmap = plt.cm.RdBu_r.with_extremes(bad="white")
        vmax = np.nanpercentile(np.abs(finite_vals), 95) if len(finite_vals) else 1
        vmax = max(vmax, 0.5)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    n_cat = len(categories)
    n_reg = len(regions)
    fig_w = max(10, n_reg * 0.8)
    fig_h = max(4, n_cat * 0.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(log_fe.values, aspect="auto", cmap=cmap, norm=norm)

    # asterisks only if not sig_only
    if not sig_only:
        for i in range(n_cat):
            for j in range(n_reg):
                fe_val = fe_pivot.iloc[i, j]
                if pd.isna(fe_val):
                    continue
                q_val = qe_pivot.iloc[i, j] if fe_val >= 1 else qd_pivot.iloc[i, j]
                if pd.notna(q_val) and q_val < Q_THRESH:
                    ax.text(j, i, "*", ha="center", va="center",
                            fontsize=10, color="black", fontweight="bold")

    ax.set_xticks(range(n_reg))
    ax.set_xticklabels([REGION_LABELS.get(r, r) for r in regions],
                       rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n_cat))
    ax.set_yticklabels(categories, fontsize=10)
    ax.set_xlabel("Region", fontsize=12)
    ax.set_ylabel("Functional category", fontsize=12)
    ax.set_title(f"{direction.capitalize()}\n{log_label}", fontsize=13)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(log_label, fontsize=10)

    plt.tight_layout()

    fname = f"heatmap_category_{direction}_{log_name}{clip_tag}{sig_tag}.png"
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")

# ── generate 2 heatmaps ────────────────────────────────────────────────────
for direction in ["strengthening", "weakening"]:
    make_heatmap(df, direction, log_fn, log_label,
                 OUT_DIR, args.log, args.clip, clip_tag, args.sig_only, sig_tag)

print("Done.")
