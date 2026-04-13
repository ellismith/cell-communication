#!/usr/bin/env python3
"""
Sex effect heatmaps for Glutamatergic_5 across all functional categories.
For each region × category with at least one significant sex effect (sex_qval < 0.05)
involving Glutamatergic_5 as sender or receiver, plot a heatmap.
Color = sex_coef (positive = higher in males, negative = higher in females).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

BASE     = Path("/Users/elliotsmith/Desktop/regression_results")
ANN_PATH = "/Users/elliotsmith/Desktop/cpdb_lr_annotations.csv"
OUT      = Path("/Users/elliotsmith/Desktop/glut5_sex_heatmaps")
OUT.mkdir(parents=True, exist_ok=True)

QVAL_THRESH = 0.05
FOCAL       = "Glutamatergic_5"
REGIONS     = ["acc", "nac", "hip", "cn", "lcb", "mb", "mdtn", "dlpfc", "ec", "ipp", "m1"]
REGION_DISPLAY = {
    "acc": "ACC", "cn": "CN", "dlpfc": "dlPFC", "ec": "EC", "hip": "HIP",
    "ipp": "IPP", "lcb": "lCB", "m1": "M1", "mb": "MB", "mdtn": "mdTN", "nac": "NAc",
}
BROAD_ORDER = ["Glutamatergic", "GABA", "Astrocyte", "Oligo", "OPC", "Microglia", "Vascular", "Other"]
BROAD_COLORS = {
    "Glutamatergic": "#e74c3c", "GABA": "#9b59b6",
    "Astrocyte": "#3498db",     "Oligo": "#2ecc71",
    "OPC": "#1abc9c",           "Microglia": "#e67e22",
    "Vascular": "#e91e63",      "Other": "#7f8c8d",
}

ann = pd.read_csv(ANN_PATH)[["lr_pair", "broad_category"]].drop_duplicates()
ann_lookup = dict(zip(ann["lr_pair"], ann["broad_category"]))

def broad_class(s):
    for b in BROAD_ORDER:
        if s.startswith(b):
            return b
    return "Other"

def sort_louvains(lst):
    def key(s):
        bc = broad_class(s)
        num = int(re.search(r'(\d+)$', s).group(1)) if re.search(r'(\d+)$', s) else 0
        return (BROAD_ORDER.index(bc) if bc in BROAD_ORDER else 99, num)
    return sorted(lst, key=key)

def parse_interaction(s):
    parts = s.split("|")
    if len(parts) < 4:
        return None
    return parts[0], parts[1], parts[2] + "|" + parts[3]

def draw_panel(ax, sub, partner_col, sig_lrs, vmax, title):
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    if sub.empty or not sig_lrs:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_axis_off()
        return

    sub = sub[sub["lr_pair"].isin(sig_lrs)]
    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_axis_off()
        return

    sig_sub = sub[sub["sex_qval"] < QVAL_THRESH]
    sig_partners = set(sig_sub[partner_col].unique())
    if not sig_partners:
        ax.text(0.5, 0.5, "No sig partners", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_axis_off()
        return

    sub = sub[sub[partner_col].isin(sig_partners)]
    partners = sort_louvains(list(sig_partners))

    mat = pd.DataFrame(np.nan, index=sig_lrs, columns=partners)
    for _, row in sub.iterrows():
        lr, p = row["lr_pair"], row[partner_col]
        if lr in sig_lrs and p in partners:
            if row["sex_qval"] < QVAL_THRESH:
                mat.loc[lr, p] = row["sex_coef"]

    mat = mat.dropna(how="all")
    if mat.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_axis_off()
        return

    sns.heatmap(mat, ax=ax, cmap=cmap, vmin=-vmax, vmax=vmax,
                linewidths=0.4, linecolor="#ccc",
                mask=mat.isna(), cbar=True,
                cbar_kws={"shrink": 0.6, "label": "sex β (M>0)"})

    ax.set_xticklabels(partners, rotation=45, ha="right", fontsize=7)
    for tick, p in zip(ax.get_xticklabels(), partners):
        tick.set_color(BROAD_COLORS.get(broad_class(p), "#333"))
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("")
    ax.set_ylabel("")

def process_region(region):
    reg_dir = BASE / f"regression_{region}"
    csv_files = list(reg_dir.glob(f"whole_{region}_age_sex_regression*.csv"))
    if not csv_files:
        print(f"  [skip] not found in {reg_dir}")
        return

    df = pd.read_csv(csv_files[0])
    parsed = df["interaction"].apply(parse_interaction)
    df = df[parsed.notna()].copy()
    parsed = parsed[parsed.notna()]
    df["sender"]   = parsed.apply(lambda x: x[0])
    df["receiver"] = parsed.apply(lambda x: x[1])
    df["lr_pair"]  = parsed.apply(lambda x: x[2])
    df["category"] = df["lr_pair"].map(ann_lookup)

    focal_df = df[(df["sender"] == FOCAL) | (df["receiver"] == FOCAL)]
    if focal_df.empty:
        print(f"  [skip] no {FOCAL} interactions")
        return

    # find categories with at least one significant sex effect
    sig_cats = focal_df.loc[focal_df["sex_qval"] < QVAL_THRESH, "category"].dropna().unique()
    if len(sig_cats) == 0:
        print(f"  [skip] no significant sex effects for {FOCAL}")
        return

    print(f"  Found {len(sig_cats)} categories with sig sex effects: {sorted(sig_cats)}")

    region_label = REGION_DISPLAY.get(region, region.upper())

    for cat in sorted(sig_cats):
        cat_df = focal_df[focal_df["category"] == cat]
        send_df = cat_df[cat_df["sender"] == FOCAL]
        recv_df = cat_df[cat_df["receiver"] == FOCAL]

        sig_lrs = sorted(
            set(send_df.loc[send_df["sex_qval"] < QVAL_THRESH, "lr_pair"]) |
            set(recv_df.loc[recv_df["sex_qval"] < QVAL_THRESH, "lr_pair"])
        )
        if not sig_lrs:
            continue

        all_coefs = cat_df.loc[
            (cat_df["lr_pair"].isin(sig_lrs)) & (cat_df["sex_qval"] < QVAL_THRESH),
            "sex_coef"
        ].dropna()
        vmax = max(np.percentile(np.abs(all_coefs), 98), 0.01) if len(all_coefs) else 0.05

        n_send = send_df[send_df["sex_qval"] < QVAL_THRESH]["receiver"].nunique()
        n_recv = recv_df[recv_df["sex_qval"] < QVAL_THRESH]["sender"].nunique()
        fig_w  = max(24, (n_send + n_recv) * 0.6 + 8)
        fig_h  = max(6, len(sig_lrs) * 0.45 + 3)

        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                                 gridspec_kw={"wspace": 0.5})
        fig.suptitle(f"{region_label} — {cat} | {FOCAL} (sex effects, M>0=red)",
                     fontsize=12, fontweight="bold", y=1.01)

        draw_panel(axes[0], send_df, "receiver", sig_lrs, vmax, f"{FOCAL} → partner")
        draw_panel(axes[1], recv_df, "sender",   sig_lrs, vmax, f"partner → {FOCAL}")

        fname = OUT / f"{region_label}_Glut5_sex_{cat.replace(' ', '_')}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"    ✓ {fname.name}")

print("=== Glutamatergic_5 sex effect heatmaps ===")
for region in REGIONS:
    print(f"\n{region}")
    process_region(region)
print("\nDone.")
