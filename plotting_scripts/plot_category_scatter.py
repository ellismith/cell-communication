#!/usr/bin/env python3
"""
Scatter plots (age vs lr_means) with regression lines for each significant
LR pair × sender→receiver combo in each category × region.
One PNG per LR pair × combo saved to:
  results/within_region_analysis/hypergeometric_all_regions/category_plots/scatter/

Usage:
  python plot_category_scatter.py
  python plot_category_scatter.py --region hip
  python plot_category_scatter.py --category "Axon guidance"
  python plot_category_scatter.py --region hip --category "Axon guidance"
  python plot_category_scatter.py --region hip --category "Glutamate signaling" --lr_pair "GLS|GRIA4"
  python plot_category_scatter.py --region hip --category "Glutamate signaling" --lr_pair "GLS|GRIA4" --sender "Astrocyte" --receiver "OPC"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import glob
import argparse

# ── args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--region",   type=str, default=None)
parser.add_argument("--category", type=str, default=None)
parser.add_argument("--lr_pair",  type=str, default=None,
                    help="Filter to specific LR pair e.g. 'GLS|GRIA4'")
parser.add_argument("--sender",   type=str, default=None,
                    help="Filter to specific broad sender cell type e.g. 'Astrocyte'")
parser.add_argument("--receiver", type=str, default=None,
                    help="Filter to specific broad receiver cell type e.g. 'OPC'")
args = parser.parse_args()

# ── paths ──────────────────────────────────────────────────────────────────
CATEGORY_DIR  = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/category_tables"
LR_MATRIX_DIR = "/scratch/easmit31/cell_cell/results/lr_matrices"
OUT_DIR       = "/scratch/easmit31/cell_cell/results/within_region_analysis/hypergeometric_all_regions/category_plots/scatter"
os.makedirs(OUT_DIR, exist_ok=True)

REGION_LABELS = {
    "acc": "ACC", "cn": "CN", "dlpfc": "dlPFC", "ec": "EC", "hip": "HIP",
    "ipp": "IPP", "lcb": "lCB", "m1": "M1", "mb": "MB", "mdtn": "mdTN", "nac": "NAc",
}

def load_lr_matrix(region):
    region_upper = region.upper()
    fpath = os.path.join(LR_MATRIX_DIR, f"{region_upper}_nothresh_minage1p0_matrix.csv")
    if not os.path.exists(fpath):
        fpath = os.path.join(LR_MATRIX_DIR, f"{region}_nothresh_minage1p0_matrix.csv")
    if not os.path.exists(fpath):
        return None, None
    df   = pd.read_csv(fpath, index_col=0)
    ages = df.loc["age"].astype(float)
    df   = df.drop(["age", "sex"], errors="ignore")
    return df, ages

# ── load category files ────────────────────────────────────────────────────
category_files = sorted(glob.glob(os.path.join(CATEGORY_DIR, "category_*.csv")))

if args.category:
    cat_fname = "category_" + args.category.replace(" ", "_").replace("/", "_") + ".csv"
    category_files = [f for f in category_files if os.path.basename(f) == cat_fname]
    if not category_files:
        print(f"ERROR: no category file found matching '{args.category}'")
        exit(1)

print(f"Found {len(category_files)} category file(s) to process")

for fpath in category_files:
    cat_name = os.path.basename(fpath).replace("category_", "").replace(".csv", "").replace("_", " ")
    df = pd.read_csv(fpath)

    # add broad cell type columns
    df["sender_broad"]   = df["sender"].str.replace(r"_\d+$", "", regex=True)
    df["receiver_broad"] = df["receiver"].str.replace(r"_\d+$", "", regex=True)

    # apply filters
    if args.region:
        df = df[df["region"] == args.region.lower()]
    if args.lr_pair:
        df = df[df["lr_pair"] == args.lr_pair]
    if args.sender:
        df = df[df["sender_broad"] == args.sender]
    if args.receiver:
        df = df[df["receiver_broad"] == args.receiver]

    if len(df) == 0:
        print(f"  WARNING: no data after filtering for {cat_name}")
        continue

    for region, rdf in df.groupby("region"):
        matrix, ages = load_lr_matrix(region)
        if matrix is None:
            print(f"  WARNING: no lr_matrix found for {region}, skipping.")
            continue

        rdf = rdf.copy()
        region_label = REGION_LABELS.get(region, region)

        subdir = os.path.join(OUT_DIR, f"{cat_name.replace(' ', '_')}_{region_label}")
        os.makedirs(subdir, exist_ok=True)

        n_saved = 0
        for _, row in rdf.iterrows():
            lr_pair  = row["lr_pair"]
            sender   = row["sender"]
            receiver = row["receiver"]

            interaction = f"{sender}|{receiver}|{lr_pair}"

            if interaction not in matrix.index:
                continue

            lr_vals = matrix.loc[interaction]
            common  = lr_vals.index.intersection(ages.index)
            x = ages[common].values.astype(float)
            y = lr_vals[common].values

            mask = ~np.isnan(y.astype(float))
            x, y = x[mask], y[mask].astype(float)

            if len(x) < 3:
                continue

            slope, intercept, r, p, _ = stats.linregress(x, y)

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(x, y, color="steelblue", alpha=0.7, edgecolors="none", s=40)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept,
                    color="#d62728" if slope > 0 else "#1f77b4", linewidth=2)

            ax.set_xlabel("Age (years)", fontsize=11)
            ax.set_ylabel("LR means", fontsize=11)
            ax.set_title(f"{lr_pair}\n{sender} → {receiver}\n"
                         f"{region_label} | β={slope:.4f}  r={r:.2f}  p={p:.3g}",
                         fontsize=9)

            plt.tight_layout()

            fname = f"{lr_pair.replace('|', '_')}_{sender}_{receiver}.png"
            fname = fname.replace("/", "_").replace(" ", "_")
            fig.savefig(os.path.join(subdir, fname), dpi=150, bbox_inches="tight")
            plt.close()
            n_saved += 1

        print(f"  Done: {cat_name} — {region_label} ({n_saved} plots saved)")

print("Done.")
