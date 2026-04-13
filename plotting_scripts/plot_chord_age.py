#!/usr/bin/env python3
"""
Circular arrow plot showing age-associated cell-cell communication.
Nodes = broad cell classes in a circle.
Arrows = sender→receiver, red=increases with age, blue=decreases.
Width = number of significant LR pairs.
Self-loops drawn as arcs outside the circle.

Usage:
    python plot_chord_age.py --region HIP
    python plot_chord_age.py --region HIP --min_sig 5
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import argparse
import re
from pathlib import Path

REGIONS = ["ACC","CN","dlPFC","EC","HIP","IPP","lCb","M1","MB","mdTN","NAc"]
BASE_DIR = Path("/scratch/easmit31/cell_cell/results/within_region_analysis_corrected")

CELLTYPE_COLORS = {
    'Astrocyte':  '#e6ab02', 'GABA':       '#7570b3',
    'Glut':       '#1b9e77', 'Microglia':  '#d95f02',
    'OPC':        '#66a61e', 'Oligo':      '#a6d854',
    'Vascular':   '#e7298a', 'Basket':     '#a6761d',
    'Cerebellar': '#666666', 'Ependymal':  '#17becf',
    'MSN':        '#bcbd22', 'Midbrain':   '#8c564b',
}

def get_class(s):
    return next((k for k in CELLTYPE_COLORS if s.startswith(k)), 'Unknown')

def get_color(c):
    return CELLTYPE_COLORS.get(c, '#aaaaaa')

parser = argparse.ArgumentParser()
parser.add_argument('--region', default=None)
parser.add_argument('--qval', type=float, default=0.05)
parser.add_argument('--min_sig', type=int, default=3)
args = parser.parse_args()

regions = [args.region] if args.region else REGIONS

for region in regions:
    csv = BASE_DIR / f"regression_{region}" / f"whole_{region.lower()}_age_sex_regression.csv"
    if not csv.exists():
        print(f"Missing: {csv}"); continue

    df = pd.read_csv(csv)
    sig = df[df['age_qval'] < args.qval].copy()
    if len(sig) == 0:
        print(f"{region}: no significant interactions"); continue

    parts = sig['interaction'].str.split('|')
    sig['sender_class']   = parts.str[0].apply(get_class)
    sig['receiver_class'] = parts.str[1].apply(get_class)

    # aggregate: n_sig and mean_coef per sender→receiver broad combo
    edges = (sig.groupby(['sender_class','receiver_class'])
               .agg(n_sig=('age_coef','count'), mean_coef=('age_coef','mean'))
               .reset_index())
    edges = edges[edges['n_sig'] >= args.min_sig]

    if len(edges) == 0:
        print(f"{region}: no edges above min_sig={args.min_sig}"); continue

    celltypes = sorted(set(edges['sender_class']) | set(edges['receiver_class']))
    n = len(celltypes)
    angles = {ct: 2 * np.pi * i / n for i, ct in enumerate(celltypes)}
    pos = {ct: (np.cos(angles[ct]), np.sin(angles[ct])) for ct in celltypes}

    NODE_RADIUS = 0.12
    max_sig = edges['n_sig'].max()
    min_lw, max_lw = 1.0, 6.0

    coef_abs_max = edges['mean_coef'].abs().max()
    norm = TwoSlopeNorm(vmin=-coef_abs_max, vcenter=0, vmax=coef_abs_max)
    cmap = plt.cm.RdBu_r

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal'); ax.axis('off')

    out_dir = BASE_DIR / f"regression_{region}" / "chord_diagrams"
    out_dir.mkdir(exist_ok=True)

    for _, row in edges.iterrows():
        src, tgt = row['sender_class'], row['receiver_class']
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        lw = min_lw + (max_lw - min_lw) * (row['n_sig'] / max_sig)
        color = cmap(norm(row['mean_coef']))

        if src == tgt:
            # self-loop
            angle = angles[src]
            loop_x = 1.35 * np.cos(angle)
            loop_y = 1.35 * np.sin(angle)
            self_loop = mpatches.Arc((loop_x, loop_y), 0.25, 0.25,
                                     angle=0, theta1=0, theta2=300,
                                     color=color, lw=lw)
            ax.add_patch(self_loop)
        else:
            # offset start/end to node edge
            dx, dy = x1 - x0, y1 - y0
            dist = np.sqrt(dx**2 + dy**2)
            ux, uy = dx/dist, dy/dist
            sx = x0 + ux * NODE_RADIUS
            sy = y0 + uy * NODE_RADIUS
            ex = x1 - ux * NODE_RADIUS
            ey = y1 - uy * NODE_RADIUS

            # bezier control point (perpendicular offset)
            mid_x = (sx + ex) / 2 - 0.25 * uy
            mid_y = (sy + ey) / 2 + 0.25 * ux

            # draw bezier
            t = np.linspace(0, 1, 100)
            bx = (1-t)**2 * sx + 2*(1-t)*t * mid_x + t**2 * ex
            by = (1-t)**2 * sy + 2*(1-t)*t * mid_y + t**2 * ey
            ax.plot(bx, by, color=color, lw=lw, alpha=0.8, zorder=2)

            # arrowhead
            t2 = 0.92
            ax_pt = (1-t2)**2*sx + 2*(1-t2)*t2*mid_x + t2**2*ex
            ay_pt = (1-t2)**2*sy + 2*(1-t2)*t2*mid_y + t2**2*ey
            ax.annotate('', xy=(ex, ey), xytext=(ax_pt, ay_pt),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=lw, mutation_scale=12 + lw*2),
                        zorder=3)

    # draw nodes
    for ct in celltypes:
        x, y = pos[ct]
        circle = mpatches.Circle((x, y), NODE_RADIUS,
                                  color=get_color(ct), zorder=5,
                                  ec='white', lw=2)
        ax.add_patch(circle)
        lx = 1.22 * np.cos(angles[ct])
        ly = 1.22 * np.sin(angles[ct])
        ha = 'left' if lx > 0.1 else ('right' if lx < -0.1 else 'center')
        va = 'bottom' if ly > 0.1 else ('top' if ly < -0.1 else 'center')
        ax.text(lx, ly, ct, ha=ha, va=va, fontsize=10, fontweight='bold',
                color=get_color(ct),
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.35, pad=0.02)
    cbar.set_label('Mean Age Coefficient', fontsize=9)

    ax.set_title(f"{region} — Age-Associated Cell Communication\n"
                 f"Color = mean age_coef | Width = n sig LR pairs (q<{args.qval}, min={args.min_sig})",
                 fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    fname = out_dir / f"{region.lower()}_chord_age_minsig{args.min_sig}.png"
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"✓ {fname}")
    plt.close()

print("Done.")
