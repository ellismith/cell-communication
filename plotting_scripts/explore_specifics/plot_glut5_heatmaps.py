#!/usr/bin/env python3
"""
Two heatmaps per region for interactions involving Glutamatergic_5.
Heatmap 1: Synaptic adhesion LR pairs
Heatmap 2: Glutamate signaling LR pairs
Cols split into sender (Glut5→) and receiver (→Glut5) sections.
Color = age_coef, white where q>=0.05.
Only rows AND cols with at least one q<0.05 cell shown.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from lr_functional_annotations import LR_FUNCTIONS

BASE = Path("/scratch/easmit31/cell_cell/results/within_region_analysis")
OUT  = Path("/scratch/easmit31/cell_cell/results/glut5_heatmaps")
OUT.mkdir(parents=True, exist_ok=True)

QVAL_THRESH = 0.05
FOCAL       = "Glutamatergic_5"
REGIONS     = ["ACC","NAc","HIP","CN","lCb","MB","mdTN","dlPFC","EC","IPP","M1"]
BROAD_ORDER = ["Glutamatergic","GABA","Astrocyte","Oligo","OPC","Microglia","Vascular","Other"]

BROAD_COLORS = {
    "Glutamatergic": "#e74c3c", "GABA": "#9b59b6",
    "Astrocyte": "#3498db",     "Oligo": "#2ecc71",
    "OPC": "#1abc9c",           "Microglia": "#e67e22",
    "Vascular": "#e91e63",      "Other": "#7f8c8d",
}

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

    # Only keep columns with at least one significant cell
    sig_sub = sub[sub["age_qval"] < QVAL_THRESH]
    sig_partners = set(sig_sub[partner_col].unique())
    sub = sub[sub[partner_col].isin(sig_partners)]
    partners = sort_louvains(list(sig_partners))

    mat  = pd.DataFrame(np.nan, index=sig_lrs, columns=partners)
    for _, row in sub.iterrows():
        lr, p = row["lr_pair"], row[partner_col]
        if lr in sig_lrs and p in partners:
            if row["age_qval"] < QVAL_THRESH:
                mat.loc[lr, p] = row["age_coef"]

    # Drop rows that ended up all-NaN after col filtering
    mat = mat.dropna(how="all")
    if mat.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_axis_off()
        return

    sns.heatmap(mat, ax=ax, cmap=cmap, vmin=-vmax, vmax=vmax,
                linewidths=0.4, linecolor="#ccc",
                mask=mat.isna(), cbar=True,
                cbar_kws={"shrink": 0.6, "label": "age β"})

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    for tick, p in zip(ax.get_xticklabels(), partners):
        tick.set_color(BROAD_COLORS.get(broad_class(p), "#333"))
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("")
    ax.set_ylabel("")

def process_region(region):
    csv = BASE / f"regression_{region}" / f"whole_{region.lower()}_age_sex_regression.csv"
    if not csv.exists():
        print(f"  [skip] not found: {csv.name}")
        return

    df = pd.read_csv(csv)
    parsed = df["interaction"].apply(parse_interaction)
    df = df[parsed.notna()].copy()
    parsed = parsed[parsed.notna()]
    df["sender"]   = parsed.apply(lambda x: x[0])
    df["receiver"] = parsed.apply(lambda x: x[1])
    df["lr_pair"]  = parsed.apply(lambda x: x[2])
    df["category"] = df["lr_pair"].map(LR_FUNCTIONS)

    focal_df = df[(df["sender"] == FOCAL) | (df["receiver"] == FOCAL)]
    if focal_df.empty:
        print(f"  [skip] no {FOCAL} interactions")
        return

    for cat in ["Synaptic adhesion", "Glutamate signaling"]:
        cat_df = focal_df[focal_df["category"] == cat]
        if cat_df.empty:
            print(f"  [skip] no {cat} pairs")
            continue

        send_df = cat_df[cat_df["sender"] == FOCAL]
        recv_df = cat_df[cat_df["receiver"] == FOCAL]

        sig_lrs = sorted(
            set(send_df.loc[send_df["age_qval"] < QVAL_THRESH, "lr_pair"]) |
            set(recv_df.loc[recv_df["age_qval"] < QVAL_THRESH, "lr_pair"])
        )
        if not sig_lrs:
            print(f"  [skip] no significant pairs for {cat} in {region}")
            continue

        all_coefs = cat_df.loc[
            (cat_df["lr_pair"].isin(sig_lrs)) & (cat_df["age_qval"] < QVAL_THRESH),
            "age_coef"
        ].dropna()
        vmax = max(np.percentile(np.abs(all_coefs), 98), 0.01) if len(all_coefs) else 0.05

        fig, axes = plt.subplots(1, 2, figsize=(18, max(4, len(sig_lrs) * 0.4 + 2)),
                                 gridspec_kw={"wspace": 0.5})
        fig.suptitle(f"{region} — {cat} | {FOCAL}",
                     fontsize=12, fontweight="bold", y=1.01)

        draw_panel(axes[0], send_df, "receiver", sig_lrs, vmax, f"{FOCAL} → partner")
        draw_panel(axes[1], recv_df, "sender",   sig_lrs, vmax, f"partner → {FOCAL}")

        fname = cat.replace(" ", "_")
        out = OUT / f"{region}_Glut5_{fname}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {out.name}")

print("=== Glutamatergic_5 heatmaps ===")
for region in REGIONS:
    print(f"\n{region}")
    process_region(region)
print("\nDone.")
