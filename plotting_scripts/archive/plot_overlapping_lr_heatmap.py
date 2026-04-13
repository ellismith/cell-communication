#!/usr/bin/env python3
"""
For cases where the same LR pair appears in both strengthening and weakening
for the same broad sender→receiver combo, plot a heatmap:
  - One figure per case (region × category × broad combo)
  - One subplot per LR pair
  - rows = Louvain senders, cols = Louvain receivers
  - color = age_coef (red=strengthening, blue=weakening, grey=not sig)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

CATEGORY_DIR = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/category_tables"
OUT_DIR      = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/forest_plots"
os.makedirs(OUT_DIR, exist_ok=True)

cases = [
    ("lcb",   "Glutamate signaling", "Cerebellar",    "Cerebellar",
     ["SLC1A6|GRIN2A", "SLC1A6|GRIN2C"]),
    ("dlpfc", "Glutamate signaling", "Glutamatergic", "Glutamatergic",
     ["GLS2|GRIA1", "SLC1A3|GRIA1", "SLC1A6|GRIA1", "SLC1A6|GRIA4"]),
    ("m1",    "Synaptic adhesion",   "Glutamatergic", "Glutamatergic",
     ["NLGN1|NRXN2", "SLIT1|ROBO1", "SLIT2|ROBO1", "SLIT2|ROBO2"]),
]

REGION_LABELS = {
    "acc": "ACC", "cn": "CN", "dlpfc": "dlPFC", "ec": "EC", "hip": "HIP",
    "ipp": "IPP", "lcb": "lCB", "m1": "M1", "mb": "MB", "mdtn": "mdTN", "nac": "NAc",
}

for region, category, sender_broad, receiver_broad, lr_pairs in cases:
    fname = category.replace(" ", "_")
    df    = pd.read_csv(os.path.join(CATEGORY_DIR, f"category_{fname}.csv"))
    df["sender_broad"]   = df["sender"].str.replace(r"_\d+$","",regex=True)
    df["receiver_broad"] = df["receiver"].str.replace(r"_\d+$","",regex=True)

    sub = df[
        (df["region"] == region) &
        (df["sender_broad"] == sender_broad) &
        (df["receiver_broad"] == receiver_broad) &
        (df["lr_pair"].isin(lr_pairs))
    ].copy()

    # get all unique senders and receivers across all LR pairs
    all_senders   = sorted(sub["sender"].unique(),
                           key=lambda x: int(x.split("_")[-1]))
    all_receivers = sorted(sub["receiver"].unique(),
                           key=lambda x: int(x.split("_")[-1]))

    n_lr  = len(lr_pairs)
    ncols = min(n_lr, 2)
    nrows = int(np.ceil(n_lr / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * max(6, len(all_receivers) * 0.4),
                                      nrows * max(5, len(all_senders) * 0.4)))
    axes = np.array(axes).flatten()

    # global vmax across all LR pairs
    vmax = np.nanpercentile(np.abs(sub["age_coef"].values), 95)
    vmax = max(vmax, 0.001)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r.with_extremes(bad="lightgrey")

    for i, lr_pair in enumerate(lr_pairs):
        ax     = axes[i]
        lr_sub = sub[sub["lr_pair"] == lr_pair]

        # pivot: rows=sender, cols=receiver, values=age_coef
        pivot = lr_sub.pivot_table(
            index="sender", columns="receiver",
            values="age_coef", aggfunc="mean"
        )
        # reindex to all senders/receivers for consistent axes
        pivot = pivot.reindex(index=all_senders, columns=all_receivers)

        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)

        ax.set_xticks(range(len(all_receivers)))
        ax.set_xticklabels(
            [r.split("_")[-1] for r in all_receivers],
            rotation=90, fontsize=7
        )
        ax.set_yticks(range(len(all_senders)))
        ax.set_yticklabels(
            [s.split("_")[-1] for s in all_senders],
            fontsize=7
        )
        ax.set_xlabel(f"{receiver_broad} subcluster", fontsize=8)
        ax.set_ylabel(f"{sender_broad} subcluster", fontsize=8)
        ax.set_title(lr_pair, fontsize=9, fontweight="bold")

        fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02,
                     label="age_coef")

    # hide unused axes
    for j in range(len(lr_pairs), len(axes)):
        axes[j].axis("off")

    region_label = REGION_LABELS.get(region, region)
    fig.suptitle(
        f"{region_label} | {category}\n"
        f"{sender_broad} → {receiver_broad}\n"
        f"(red=strengthening, blue=weakening, grey=not sig)",
        fontsize=11, y=1.01
    )

    plt.tight_layout()

    fname_out = f"heatmap_louvain_{region}_{category.replace(' ','_')}_{sender_broad}_{receiver_broad}.png"
    fpath     = os.path.join(OUT_DIR, fname_out)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname_out}")

print("Done.")
