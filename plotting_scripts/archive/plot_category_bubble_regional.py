#!/usr/bin/env python3
"""
Bubble plot per functional category showing LR pairs × regions,
grouped by sender or receiver cell type.

  y = LR pair
  x = region, grouped by sender (or receiver) cell type
  within each group, separate columns per receiver (or sender)
  color = mean age_coef across significant Louvain combos
  size = n significant Louvain combos

Two plots per category: strengthening and weakening.

Usage:
  python plot_category_bubble_regional.py --group_by sender
  python plot_category_bubble_regional.py --group_by receiver
  python plot_category_bubble_regional.py --group_by sender --category "Glutamate signaling"
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
parser.add_argument("--group_by", choices=["sender", "receiver"], default="sender")
parser.add_argument("--category", type=str, default=None)
args = parser.parse_args()

# ── paths ──────────────────────────────────────────────────────────────────
CATEGORY_DIR = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/category_tables"
OUT_DIR      = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/category_plots/bubble_regional"
os.makedirs(OUT_DIR, exist_ok=True)

REGION_ORDER  = ["acc", "cn", "dlpfc", "ec", "hip", "ipp", "lcb", "m1", "mb", "mdtn", "nac"]
REGION_LABELS = {
    "acc": "ACC", "cn": "CN", "dlpfc": "dlPFC", "ec": "EC", "hip": "HIP",
    "ipp": "IPP", "lcb": "lCB", "m1": "M1", "mb": "MB", "mdtn": "mdTN", "nac": "NAc",
}

# ── load category files ────────────────────────────────────────────────────
category_files = sorted(glob.glob(os.path.join(CATEGORY_DIR, "category_*.csv")))
if args.category:
    cat_fname = "category_" + args.category.replace(" ", "_").replace("/", "_") + ".csv"
    category_files = [f for f in category_files if os.path.basename(f) == cat_fname]
    if not category_files:
        print(f"ERROR: no category file found matching '{args.category}'")
        exit(1)

print(f"Found {len(category_files)} category file(s), grouping by {args.group_by}")

# ── plotting function ──────────────────────────────────────────────────────
def make_bubble_plot(df, direction, cat_name, group_by, out_dir):
    sub = df[df["direction"] == direction].copy()
    if len(sub) == 0:
        return

    # collapse Louvain to broad cell types
    sub["sender_broad"]   = sub["sender"].str.replace(r"_\d+$", "", regex=True)
    sub["receiver_broad"] = sub["receiver"].str.replace(r"_\d+$", "", regex=True)

    # aggregate: mean age_coef and count per lr_pair × region × sender_broad × receiver_broad
    agg = sub.groupby(["lr_pair", "region", "sender_broad", "receiver_broad"]).agg(
        mean_coef=("age_coef", "mean"),
        n_combos=("age_coef", "count")
    ).reset_index()

    group_col  = "sender_broad"   if group_by == "sender" else "receiver_broad"
    other_col  = "receiver_broad" if group_by == "sender" else "sender_broad"
    group_label = "Sender"  if group_by == "sender" else "Receiver"
    other_label = "Receiver" if group_by == "sender" else "Sender"

    groups   = sorted(agg[group_col].unique())
    regions  = [r for r in REGION_ORDER if r in agg["region"].unique()]
    lr_pairs = sorted(agg["lr_pair"].unique())
    y_pos    = {lr: i for i, lr in enumerate(lr_pairs)}

    # build x positions
    x = 0
    x_pos            = {}
    x_tick_pos       = []
    x_tick_labels    = []
    group_boundaries = {}
    group_label_pos  = {}

    for group in groups:
        start_x    = x
        group_data = agg[agg[group_col] == group]
        for region in regions:
            region_others = sorted(
                group_data[group_data["region"] == region][other_col].unique()
            )
            if not region_others:
                continue
            for other in region_others:
                x_pos[(group, region, other)] = x
                x_tick_pos.append(x)
                x_tick_labels.append(
                    f"{other} | {REGION_LABELS.get(region, region)}"
                )
                x += 1
            x += 0.5  # small gap between regions within a group
        group_boundaries[group] = (start_x, x - 1)
        group_label_pos[group]  = (start_x + x - 1) / 2
        x += 1.5  # larger gap between groups

    if not x_pos:
        return

    # colormap
    vmax = np.percentile(np.abs(agg["mean_coef"].values), 95)
    vmax = max(vmax, 0.001)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r

    max_n      = agg["n_combos"].max()
    size_scale = 200 / max_n

    fig_w = max(14, len(x_tick_pos) * 0.3)
    fig_h = max(6,  len(lr_pairs)   * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # shaded group backgrounds and headers
    colors_bg = plt.cm.tab20.colors
    for i, group in enumerate(groups):
        sx, ex = group_boundaries[group]
        ax.axvspan(sx - 0.75, ex + 0.75, alpha=0.07,
                   color=colors_bg[i % len(colors_bg)])
        ax.text(group_label_pos[group], len(lr_pairs) + 0.3, group,
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=colors_bg[i % len(colors_bg)])

    # draw bubbles
    for _, row in agg.iterrows():
        key = (row[group_col], row["region"], row[other_col])
        if key not in x_pos:
            continue
        xi    = x_pos[key]
        yi    = y_pos[row["lr_pair"]]
        size  = max(row["n_combos"] * size_scale, 10)
        color = cmap(norm(row["mean_coef"]))
        ax.scatter(xi, yi, s=size, c=[color], alpha=0.85,
                   edgecolors="grey", linewidths=0.3)

    # axes
    ax.set_xticks(x_tick_pos)
    ax.set_xticklabels(x_tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(lr_pairs)))
    ax.set_yticklabels(lr_pairs, fontsize=7)
    ax.set_xlabel(f"{group_label} group  →  {other_label} | Region", fontsize=11)
    ax.set_ylabel("LR pair", fontsize=11)
    ax.set_title(
        f"{cat_name} — {direction}  (grouped by {group_by})\n"
        f"size = n Louvain combos,  color = mean age_coef",
        fontsize=11
    )
    ax.set_xlim(-1, max(x_tick_pos) + 1)
    ax.set_ylim(-0.5, len(lr_pairs) + 0.8)
    ax.grid(True, linestyle="--", alpha=0.15)

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.35, pad=0.01)
    cbar.set_label("mean age_coef", fontsize=9)

    # size legend
    legend_ns = [1, max(1, max_n // 2), max_n]
    for n in sorted(set(legend_ns)):
        ax.scatter([], [], s=n * size_scale, c="grey", alpha=0.7, label=f"n={n}")
    ax.legend(title="n Louvain combos", loc="lower right", fontsize=7, framealpha=0.8)

    plt.tight_layout()

    fname = f"bubble_regional_{cat_name.replace(' ', '_')}_{direction}_groupby{group_by}.png"
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

# ── main loop ──────────────────────────────────────────────────────────────
for fpath in category_files:
    cat_name = os.path.basename(fpath).replace("category_", "").replace(".csv", "").replace("_", " ")
    df = pd.read_csv(fpath)
    print(f"\nProcessing: {cat_name}")
    for direction in ["strengthening", "weakening"]:
        make_bubble_plot(df, direction, cat_name, args.group_by, OUT_DIR)

print("\nDone.")
