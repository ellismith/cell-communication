#!/usr/bin/env python3
"""
Forest plots of significant age-associated LR pairs.
One plot per sender (or receiver) cell class, organized by the other axis.

Usage:
    python plot_age_forest.py --region HIP --split sender
    python plot_age_forest.py --region HIP --split receiver
    python plot_age_forest.py --region HIP --split sender --qval 0.1
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import argparse
import re

REGIONS = ["ACC","CN","dlPFC","EC","HIP","IPP","lCb","M1","MB","mdTN","NAc"]
BASE_DIR = Path("/scratch/easmit31/cell_cell/results/within_region_analysis_corrected")

CELLTYPE_COLORS = {
    'Astrocyte':'#e6ab02','GABA':'#7570b3','Glut':'#1b9e77',
    'Microglia':'#d95f02','OPC':'#66a61e','Oligo':'#a6d854',
    'Vascular':'#e7298a','Basket':'#a6761d','Cerebellar':'#666666',
    'Ependymal':'#17becf','MSN':'#bcbd22','Midbrain':'#8c564b','Unknown':'#aaaaaa',
}

def get_class(s):
    return next((k for k in CELLTYPE_COLORS if s.startswith(k)), 'Unknown')

def get_color(c):
    return next((CELLTYPE_COLORS[k] for k in CELLTYPE_COLORS if c.startswith(k)), '#aaaaaa')

parser = argparse.ArgumentParser()
parser.add_argument('--region', default=None)
parser.add_argument('--split', choices=['sender','receiver'], default='sender')
parser.add_argument('--qval', type=float, default=0.05)
args = parser.parse_args()

regions = [args.region] if args.region else REGIONS

for region in regions:
    csv = BASE_DIR / f"regression_{region}" / f"whole_{region.lower()}_age_sex_regression.csv"
    if not csv.exists():
        print(f"Missing: {csv}"); continue

    df = pd.read_csv(csv)
    sig = df[df['age_qval'] < args.qval].copy()
    print(f"{region}: {len(sig)} significant")
    if len(sig) == 0:
        continue

    parts = sig['interaction'].str.split('|')
    sig['sender']         = parts.str[0]
    sig['receiver']       = parts.str[1]
    sig['lr_pair']        = parts.str[2] + '→' + parts.str[3]
    sig['sender_class']   = sig['sender'].apply(get_class)
    sig['receiver_class'] = sig['receiver'].apply(get_class)

    out_dir = BASE_DIR / f"regression_{region}" / f"forest_plots_{args.split}"
    out_dir.mkdir(exist_ok=True)

    primary_col   = 'sender_class'   if args.split == 'sender' else 'receiver_class'
    secondary_col = 'receiver_class' if args.split == 'sender' else 'sender_class'
    primary_label = 'outgoing'       if args.split == 'sender' else 'incoming'
    sec_label     = 'Receiver'       if args.split == 'sender' else 'Sender'

    for primary_class, grp in sig.groupby(primary_col):
        sec_order = sorted(grp[secondary_col].unique())
        grp = grp.copy()
        grp['_sec_idx'] = grp[secondary_col].map({s:i for i,s in enumerate(sec_order)})
        grp = grp.sort_values(['_sec_idx', secondary_col, 'lr_pair'])

        n = len(grp)
        betas  = grp['age_coef'].tolist()
        stes   = grp['age_stderr'].tolist()
        colors = [get_color(c) for c in grp[secondary_col]]
        row_labels = (grp['lr_pair'] + '  [' + grp['sender'] + ' → ' + grp['receiver'] + ']').tolist()

        fig_h = max(5, n * 0.35 + 2)
        fig, ax = plt.subplots(figsize=(11, fig_h))

        for i, (beta, ste, col) in enumerate(zip(betas, stes, colors)):
            ax.plot([beta - 1.96*ste, beta + 1.96*ste], [i, i],
                    color=col, lw=1.4, alpha=0.8, zorder=2)
            ax.scatter(beta, i, color=col, s=40, zorder=3,
                       edgecolors='white', linewidths=0.4)

        ax.axvline(0, color='black', lw=0.8, linestyle='--', zorder=1)
        ax.set_ylim(-0.5, n - 0.5)
        ax.invert_yaxis()
        ax.set_yticks(range(n))
        ax.set_yticklabels(row_labels, fontsize=6.5)
        ax.set_xlabel('Age Coefficient (β)', fontsize=11)
        ax.set_title(
            f"{region} — {primary_class} {primary_label}\n"
            f"Age-associated LR pairs (q<{args.qval})  n={n}",
            fontsize=11, fontweight='bold')

        inc_patch = mpatches.Patch(color='#c0392b', label='Increases with age')
        dec_patch = mpatches.Patch(color='#2980b9', label='Decreases with age')
        sec_patches = [mpatches.Patch(color=get_color(s), label=s) for s in sec_order]
        ax.legend(handles=[inc_patch, dec_patch] + sec_patches,
                  loc='lower right', fontsize=7, framealpha=0.8,
                  title=sec_label, title_fontsize=7)

        plt.tight_layout()
        fname = f"{region.lower()}_{primary_class.lower()}_{primary_label}_forest.png"
        plt.savefig(out_dir / fname, dpi=200, bbox_inches='tight')
        print(f"  ✓ {fname}")
        plt.close()

print("Done.")
