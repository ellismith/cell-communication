#!/usr/bin/env python3
"""
For cases where the same LR pair appears in both strengthening and weakening
for the same broad sender→receiver combo, plot a strip/forest plot:
  - One figure per case (region × category × broad combo)
  - One row per LR pair
  - Each dot = one Louvain sender→receiver combo
  - x-axis = age_coef
  - color = red (strengthening) or blue (weakening)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

CATEGORY_DIR = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/category_tables"
OUT_DIR      = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/forest_plots"
os.makedirs(OUT_DIR, exist_ok=True)

STR_COLOR = "#d62728"
WK_COLOR  = "#1f77b4"

cases = [
    ("lcb",   "Glutamate signaling", "Cerebellar",    "Cerebellar",    ["SLC1A6|GRIN2A", "SLC1A6|GRIN2C"]),
    ("dlpfc", "Glutamate signaling", "Glutamatergic", "Glutamatergic", ["GLS2|GRIA1", "SLC1A3|GRIA1", "SLC1A6|GRIA1", "SLC1A6|GRIA4"]),
    ("m1",    "Synaptic adhesion",   "Glutamatergic", "Glutamatergic", ["NLGN1|NRXN2", "SLIT1|ROBO1", "SLIT2|ROBO1", "SLIT2|ROBO2"]),
]

REGION_LABELS = {
    "acc": "ACC", "cn": "CN", "dlpfc": "dlPFC", "ec": "EC", "hip": "HIP",
    "ipp": "IPP", "lcb": "lCB", "m1": "M1", "mb": "MB", "mdtn": "mdTN", "nac": "NAc",
}

for region, category, sender, receiver, lr_pairs in cases:
    fname = category.replace(" ", "_")
    df = pd.read_csv(os.path.join(CATEGORY_DIR, f"category_{fname}.csv"))
    df["sender_broad"]   = df["sender"].str.replace(r"_\d+$","",regex=True)
    df["receiver_broad"] = df["receiver"].str.replace(r"_\d+$","",regex=True)

    sub = df[
        (df["region"] == region) &
        (df["sender_broad"] == sender) &
        (df["receiver_broad"] == receiver) &
        (df["lr_pair"].isin(lr_pairs))
    ].copy()

    n_lr   = len(lr_pairs)
    fig, ax = plt.subplots(figsize=(8, max(3, n_lr * 1.2)))

    # jitter seed for reproducibility
    rng = np.random.default_rng(42)

    for i, lr_pair in enumerate(lr_pairs):
        lr_sub = sub[sub["lr_pair"] == lr_pair]

        for _, row in lr_sub.iterrows():
            color  = STR_COLOR if row["direction"] == "strengthening" else WK_COLOR
            jitter = rng.uniform(-0.15, 0.15)
            ax.scatter(row["age_coef"], i + jitter,
                       color=color, alpha=0.7, s=40,
                       edgecolors="none", zorder=3)

    # zero line
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)

    # grid lines between LR pairs
    for i in range(n_lr - 1):
        ax.axhline(i + 0.5, color="lightgrey", linewidth=0.5, zorder=1)

    ax.set_yticks(range(n_lr))
    ax.set_yticklabels(lr_pairs, fontsize=9)
    ax.set_xlabel("Age coefficient (β)", fontsize=11)
    ax.set_ylabel("LR pair", fontsize=11)
    ax.set_title(
        f"{REGION_LABELS.get(region, region)} | {category}\n"
        f"{sender} → {receiver}\n"
        f"Each dot = one Louvain sender→receiver combo",
        fontsize=10
    )

    # legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=STR_COLOR,
               markersize=8, label="Strengthening"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=WK_COLOR,
               markersize=8, label="Weakening"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    ax.invert_yaxis()

    plt.tight_layout()

    fname_out = f"forest_{region}_{category.replace(' ','_')}_{sender}_{receiver}.png"
    fpath     = os.path.join(OUT_DIR, fname_out)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname_out}")

print("Done.")
