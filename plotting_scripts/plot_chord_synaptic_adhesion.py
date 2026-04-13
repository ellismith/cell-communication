#!/usr/bin/env python3
"""
Chord plots for significant synaptic adhesion interactions, one per region.
Optional grid mode showing all 11 regions in 3col x 4row layout.

Usage:
    python plot_chord_synaptic_adhesion.py --colorbar 0          # no colorbar
    python plot_chord_synaptic_adhesion.py --colorbar 1          # shared colorbar
    python plot_chord_synaptic_adhesion.py --grid                # 3x4 grid, per-plot colorbars
    python plot_chord_synaptic_adhesion.py --grid --colorbar 1   # 3x4 grid, shared colorbar
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--colorbar', type=int, default=-1,
                    help='1=shared colorbar, 0=no colorbar, omit=per-plot colorbar')
parser.add_argument('--grid', action='store_true',
                    help='3x4 grid of all 11 regions')
args = parser.parse_args()

REGIONS      = ["ACC","CN","dlPFC","EC","HIP","IPP","lCb","M1","MB","mdTN","NAc"]
GRID_REGIONS = ["ACC","CN","dlPFC","EC","HIP","IPP","lCb","M1","MB","mdTN","NAc"]
BASE_DIR     = Path("/scratch/easmit31/cell_cell/results/within_region_analysis_corrected")
ANN_FILE     = Path("/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv")
OUT_DIR      = BASE_DIR / "chord_plots_synaptic_adhesion"
OUT_DIR.mkdir(exist_ok=True)

CELLTYPE_COLORS = {
    'Astrocyte':  '#d53a50', 'GABA':       '#4d6aa8',
    'Glut':       '#3d9995', 'Microglia':  '#bb471c',
    'OPC':        '#fbae64', 'Oligo':      '#fb8d64',
    'Vascular':   '#d1538f', 'Basket':     '#4e379a',
    'Cerebellar': '#44839f', 'Ependymal':  '#fc6238',
    'MSN':        '#5e3c99', 'Midbrain':   '#65bfb0',
}

CELLTYPE_LABELS = {
    'Astrocyte':  'Astrocyte',    'GABA':       'GABAergic',
    'Glut':       'Glutamatergic','Microglia':  'Microglia',
    'OPC':        'OPC',          'Oligo':      'Oligodendrocyte',
    'Vascular':   'Vascular',     'Basket':     'Basket',
    'Cerebellar': 'Cerebellar',   'Ependymal':  'Ependymal',
    'MSN':        'MSN',          'Midbrain':   'Midbrain',
}

def get_class(s):
    return next((k for k in CELLTYPE_COLORS if s.startswith(k)), 'Unknown')

def get_color(c):
    return CELLTYPE_COLORS.get(c, '#aaaaaa')

ann = pd.read_csv(ANN_FILE)
ann['lr_pair'] = ann['ligand'] + '|' + ann['receptor']
syn_pairs = set(ann[ann['broad_category'] == 'Synaptic adhesion']['lr_pair'])
print(f"Synaptic adhesion LR pairs: {len(syn_pairs)}")

# load all edges
all_edges = {}
for region in REGIONS:
    csv = BASE_DIR / f"regression_{region}" / f"whole_{region.lower()}_age_sex_regression.csv"
    if not csv.exists():
        continue
    print(f"Loading {region}...")
    chunks = []
    for chunk in pd.read_csv(csv, chunksize=500000):
        parts = chunk['interaction'].str.split('|', expand=True)
        chunk['sender_class']   = parts[0].str.replace(r'_\d+$','',regex=True).apply(get_class)
        chunk['receiver_class'] = parts[1].str.replace(r'_\d+$','',regex=True).apply(get_class)
        chunk['lr_pair']        = parts[2] + '|' + parts[3]
        sub = chunk[(chunk['age_qval'] < 0.05) & (chunk['lr_pair'].isin(syn_pairs))]
        if len(sub): chunks.append(sub)
    if not chunks:
        continue
    sig = pd.concat(chunks)
    edges = (sig.groupby(['sender_class','receiver_class'])
               .agg(n_sig=('age_coef','count'), mean_coef=('age_coef','mean'))
               .reset_index())
    all_edges[region] = edges

# global colorscale
all_coefs = pd.concat(all_edges.values())['mean_coef']
global_vmax = max(all_coefs.abs().max(), 0.001)
global_norm = TwoSlopeNorm(vmin=-global_vmax, vcenter=0, vmax=global_vmax)
cmap = plt.cm.RdBu_r

def draw_on_ax(ax, region, edges, use_norm, show_cbar=False):
    celltypes = sorted(set(edges['sender_class']) | set(edges['receiver_class']))
    n = len(celltypes)
    angles = {ct: 2*np.pi*i/n for i, ct in enumerate(celltypes)}
    pos    = {ct: (np.cos(angles[ct]), np.sin(angles[ct])) for ct in celltypes}

    NODE_RADIUS = 0.12
    max_sig = edges['n_sig'].max()
    min_lw, max_lw = 1.0, 6.0

    for _, row in edges.iterrows():
        src, tgt = row['sender_class'], row['receiver_class']
        x0, y0 = pos[src]; x1, y1 = pos[tgt]
        lw = min_lw + (max_lw - min_lw) * (row['n_sig'] / max_sig)
        color = cmap(use_norm(row['mean_coef']))

        if src == tgt:
            angle = angles[src]
            ax.add_patch(mpatches.Arc((1.35*np.cos(angle), 1.35*np.sin(angle)),
                                      0.25, 0.25, angle=0, theta1=0, theta2=300,
                                      color=color, lw=lw))
        else:
            dx, dy = x1-x0, y1-y0
            dist = np.sqrt(dx**2+dy**2)
            ux, uy = dx/dist, dy/dist
            sx, sy = x0+ux*NODE_RADIUS, y0+uy*NODE_RADIUS
            ex, ey = x1-ux*NODE_RADIUS, y1-uy*NODE_RADIUS
            mid_x = (sx+ex)/2 - 0.25*uy
            mid_y = (sy+ey)/2 + 0.25*ux
            t = np.linspace(0, 1, 100)
            bx = (1-t)**2*sx + 2*(1-t)*t*mid_x + t**2*ex
            by = (1-t)**2*sy + 2*(1-t)*t*mid_y + t**2*ey
            ax.plot(bx, by, color=color, lw=lw, alpha=0.8, zorder=2)
            t2 = 0.92
            ax_pt = (1-t2)**2*sx + 2*(1-t2)*t2*mid_x + t2**2*ex
            ay_pt = (1-t2)**2*sy + 2*(1-t2)*t2*mid_y + t2**2*ey
            ax.annotate('', xy=(ex,ey), xytext=(ax_pt,ay_pt),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=lw, mutation_scale=12+lw*2), zorder=3)

    for ct in celltypes:
        x, y = pos[ct]
        ax.add_patch(mpatches.Circle((x,y), NODE_RADIUS, color=get_color(ct),
                                      zorder=5, ec='white', lw=2))
        lx = 1.22*np.cos(angles[ct]); ly = 1.22*np.sin(angles[ct])
        ha = 'left' if lx>0.1 else ('right' if lx<-0.1 else 'center')
        va = 'bottom' if ly>0.1 else ('top' if ly<-0.1 else 'center')
        label = CELLTYPE_LABELS.get(ct, ct)
        ax.text(lx, ly, label, ha=ha, va=va, fontsize=7, fontweight='bold',
                color=get_color(ct),
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(region, fontsize=13, fontweight='bold', pad=4)

    if show_cbar:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(ax, width="4%", height="40%", loc='lower right',
                         bbox_to_anchor=(0.12, 0.02, 1, 1),
                         bbox_transform=ax.transAxes, borderpad=0)
        sm = ScalarMappable(cmap=cmap, norm=use_norm); sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label('Age Effect', fontsize=7)
        cbar.ax.tick_params(labelsize=6)

# ── GRID MODE ──────────────────────────────────────────────────────────────
if args.grid:
    ncols = 3
    nrows = 4
    n_regions = len(GRID_REGIONS)

    if args.colorbar == 1:
        fig = plt.figure(figsize=(ncols*7 + 1.2, nrows*6))
        gs  = fig.add_gridspec(nrows, ncols+1,
                                width_ratios=[7]*ncols + [0.6],
                                hspace=0.05, wspace=0.02)
        axes = [fig.add_subplot(gs[r, c]) for r in range(nrows) for c in range(ncols)]
        cbar_ax = fig.add_subplot(gs[:, ncols])
    else:
        fig = plt.figure(figsize=(ncols*7, nrows*6))
        gs  = fig.add_gridspec(nrows, ncols, hspace=0.05, wspace=0.02)
        axes = [fig.add_subplot(gs[r, c]) for r in range(nrows) for c in range(ncols)]
        cbar_ax = None

    for i, ax in enumerate(axes):
        if i >= n_regions:
            ax.axis('off')
            continue
        region = GRID_REGIONS[i]
        if region not in all_edges:
            ax.axis('off'); continue
        if args.colorbar == 1:
            use_norm = global_norm
            show_cbar = False
        elif args.colorbar == 0:
            use_norm = global_norm
            show_cbar = False
        else:
            coef_abs_max = max(all_edges[region]['mean_coef'].abs().max(), 0.001)
            use_norm = TwoSlopeNorm(vmin=-coef_abs_max, vcenter=0, vmax=coef_abs_max)
            show_cbar = True
        draw_on_ax(ax, region, all_edges[region], use_norm, show_cbar=show_cbar)

    if args.colorbar == 1 and cbar_ax is not None:
        sm = ScalarMappable(cmap=cmap, norm=global_norm); sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Age Effect', fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle('Synaptic Adhesion Age Effects', fontsize=14, fontweight='bold', y=1.002)
    suffix = f"_colorbar{args.colorbar}" if args.colorbar in [0,1] else "_percbar"
    fname = OUT_DIR / f"grid_chord_synaptic_adhesion_11regions{suffix}.png"
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ {fname.name}")

# ── INDIVIDUAL PLOTS ───────────────────────────────────────────────────────
else:
    for region, edges in all_edges.items():
        if args.colorbar == 1:
            use_norm = global_norm
        elif args.colorbar == 0:
            use_norm = global_norm
        else:
            coef_abs_max = max(edges['mean_coef'].abs().max(), 0.001)
            use_norm = TwoSlopeNorm(vmin=-coef_abs_max, vcenter=0, vmax=coef_abs_max)

        fig, ax = plt.subplots(figsize=(9, 9))
        draw_on_ax(ax, region, edges, use_norm, show_cbar=False)

        if args.colorbar != 0:
            sm = ScalarMappable(cmap=cmap, norm=use_norm); sm.set_array([])
            plt.colorbar(sm, ax=ax, shrink=0.35, pad=0.02).set_label('Age Effect', fontsize=9)

        ax.set_title(f"{region} — Synaptic Adhesion\n"
                     f"Color=mean age effect | Width=n sig LR pairs (q<0.05)",
                     fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout()
        fname = OUT_DIR / f"{region.lower()}_chord_synaptic_adhesion.png"
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✓ {fname.name}")

print("Done.")
