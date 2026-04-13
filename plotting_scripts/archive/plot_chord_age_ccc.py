#!/usr/bin/env python3
"""
plot_chord_age_ccc.py
Circular arrow plots showing age-related CCC.
- Nodes = cell types arranged in circle, colored by cell type
- Arrows = directed sender->receiver, width=n sig LR pairs
- Arrow color = mean age effect direction (red=increasing, blue=decreasing with age)
- Three versions: per region, global vs specific, by functional category
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--qval_thresh', type=float, default=0.05)
parser.add_argument('--min_animals', type=int, default=10)
parser.add_argument('--min_sig', type=int, default=5)
args = parser.parse_args()

BASE     = '/scratch/easmit31/cell_cell/results/within_region_analysis'
ANNOT    = '/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv'
PLOT_DIR = '/scratch/easmit31/cell_cell/results/regression_plots/chord_diagrams'
os.makedirs(PLOT_DIR, exist_ok=True)

REGIONS = ['ACC','CN','dlPFC','EC','HIP','IPP','lCb','M1','MB','mdTN','NAc']
CTS = ['Astrocyte','GABA','Glutamatergic','Microglia','Oligo','OPC',
       'Vascular','Basket','Cerebellar','Ependymal','Midbrain','MSN']

CT_COLORS = {
    'Astrocyte':     '#e6194b',
    'GABA':          '#3cb44b',
    'Glutamatergic': '#4363d8',
    'Microglia':     '#f58231',
    'Oligo':         '#911eb4',
    'OPC':           '#42d4f4',
    'Vascular':      '#f032e6',
    'Basket':        '#bfef45',
    'Cerebellar':    '#fabed4',
    'Ependymal':     '#469990',
    'Midbrain':      '#dcbeff',
    'MSN':           '#9a6324',
}

CAT_COLORS = {
    'GABA signaling':         '#3cb44b',
    'Glutamate signaling':    '#4363d8',
    'ECM/Adhesion':           '#e6194b',
    'Synaptic adhesion':      '#f032e6',
    'Neurotrophic':           '#f58231',
    'Growth factor signaling':'#42d4f4',
    'Axon guidance':          '#bfef45',
    'Monoamine signaling':    '#911eb4',
    'Wnt signaling':          '#9a6324',
    'Neuropeptide signaling': '#469990',
    'Lipid/Steroid':          '#dcbeff',
    'Immune/Inflammatory':    '#fabed4',
    'Teneurin/Latrophilin':   '#fffac8',
    'Metabolic/Other':        '#aaffc3',
    'Notch/Hedgehog':         '#808000',
    'Cholinergic signaling':  '#ffd8b1',
    'Gap junction':           '#000075',
    'Glycine signaling':      '#a9a9a9',
    'Unknown':                '#cccccc',
}

annot = pd.read_csv(ANNOT)[['lr_pair','broad_category']].drop_duplicates('lr_pair')

def draw_circular_arrow_plot(ax, connections, active_cts, title,
                              color_by='direction', max_width=8):
    """
    connections: list of (sender, receiver, n_sig, mean_beta, extra)
    active_cts: list of cell types to show
    color_by: 'direction' (red/blue) or 'category' (categorical)
    """
    n = len(active_cts)
    if n == 0:
        ax.set_title(title, fontsize=8)
        ax.axis('off')
        return

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=8, pad=4)

    # node positions on circle
    angles = {ct: 2 * np.pi * i / n - np.pi/2
              for i, ct in enumerate(active_cts)}
    r_node = 1.15
    node_r = 0.08

    # draw arrows first (behind nodes)
    max_n = max([c[2] for c in connections]) if connections else 1
    for sender, receiver, n_sig, mean_beta, extra in connections:
        if sender not in angles or receiver not in angles:
            continue
        if n_sig < args.min_sig:
            continue

        a1 = angles[sender]
        a2 = angles[receiver]
        x1 = r_node * np.cos(a1)
        y1 = r_node * np.sin(a1)
        x2 = r_node * np.cos(a2)
        y2 = r_node * np.sin(a2)

        # arrow width proportional to n_sig
        lw = max(0.5, (n_sig / max_n) * max_width)

        # color
        if color_by == 'direction':
            if mean_beta > 0:
                # red for increasing, intensity by magnitude
                intensity = min(1.0, abs(mean_beta) / 0.003)
                color = plt.cm.Reds(0.4 + 0.6 * intensity)
            else:
                intensity = min(1.0, abs(mean_beta) / 0.003)
                color = plt.cm.Blues(0.4 + 0.6 * intensity)
        else:
            color = CAT_COLORS.get(extra, '#cccccc')

        # draw curved arrow using FancyArrowPatch
        # control point toward center for curve
        ctrl_x = (x1 + x2) * 0.3
        ctrl_y = (y1 + y2) * 0.3

        # start/end slightly inside node edge
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx**2 + dy**2) + 1e-9
        sx = x1 + node_r * dx/dist
        sy = y1 + node_r * dy/dist
        ex = x2 - node_r * dx/dist
        ey = y2 - node_r * dy/dist

        arrow = FancyArrowPatch(
            (sx, sy), (ex, ey),
            connectionstyle=f'arc3,rad=0.3',
            arrowstyle=f'->,head_width={lw*0.15},head_length={lw*0.1}',
            color=color,
            linewidth=lw,
            alpha=0.75,
            zorder=2
        )
        ax.add_patch(arrow)

    # draw nodes on top
    for ct in active_cts:
        a = angles[ct]
        x = r_node * np.cos(a)
        y = r_node * np.sin(a)
        circle = plt.Circle((x, y), node_r, color=CT_COLORS[ct],
                            zorder=3, ec='white', linewidth=1)
        ax.add_patch(circle)

        # label
        lx = 1.35 * np.cos(a)
        ly = 1.35 * np.sin(a)
        ha = 'left' if np.cos(a) > 0.1 else ('right' if np.cos(a) < -0.1 else 'center')
        ax.text(lx, ly, ct, ha=ha, va='center', fontsize=5.5, fontweight='bold',
                color=CT_COLORS[ct],
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading regression data...")
region_data = {}
for region in REGIONS:
    rlow = region.lower()
    path = f'{BASE}/regression_{region}/whole_{rlow}_age_sex_regression.csv'
    if not os.path.exists(path): continue
    df = pd.read_csv(path, engine='c')
    df = df[df['n_animals'] >= args.min_animals]
    parts = df['interaction'].str.split('|', expand=True)
    df['source_ct'] = parts[0].str.rsplit('_', n=1).str[0]
    df['target_ct'] = parts[1].str.rsplit('_', n=1).str[0]
    df['lr_pair']   = parts[2] + '|' + parts[3]
    df = df[df['source_ct'].isin(CTS) & df['target_ct'].isin(CTS)]
    df = df.merge(annot, on='lr_pair', how='left')
    df['broad_category'] = df['broad_category'].fillna('Unknown')
    region_data[region] = df
    print(f'  {region}: {len(df):,} rows')

# cross-region counts for global vs specific
all_sig = pd.concat([
    df[df['age_qval'] < args.qval_thresh][['lr_pair','source_ct','target_ct']].assign(region=r)
    for r, df in region_data.items()
], ignore_index=True)
lr_region_counts = all_sig.groupby(['lr_pair','source_ct','target_ct'])['region'].nunique()

def get_connections(df, sig_mask, color_by='direction'):
    sig = df[sig_mask]
    connections = []
    for (sender, receiver), grp in sig.groupby(['source_ct','target_ct']):
        n_sig = grp['lr_pair'].nunique()
        mean_beta = grp['age_coef'].mean()
        if color_by == 'category':
            extra = grp['broad_category'].value_counts().index[0]
        else:
            extra = None
        connections.append((sender, receiver, n_sig, mean_beta, extra))
    return connections

# ── PLOT 1: Per-region (one plot per region, larger) ──────────────────────────
print("Plotting Type 1: Per-region...")
ncols = 4
nrows = int(np.ceil(len(region_data) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))
axes = axes.flatten()

for idx, (region, df) in enumerate(region_data.items()):
    conns = get_connections(df, df['age_qval'] < args.qval_thresh)
    active = sorted(set([c[0] for c in conns] + [c[1] for c in conns]))
    draw_circular_arrow_plot(axes[idx], conns, active, region)

for idx in range(len(region_data), len(axes)):
    axes[idx].axis('off')

# shared legends
fig.text(0.5, 1.01, f'Age-related CCC per region\nArrow width=n sig LR pairs, '
         f'Red=increasing / Blue=decreasing with age (qval<{args.qval_thresh})',
         ha='center', fontsize=11, fontweight='bold')

ct_legend = [mpatches.Patch(facecolor=CT_COLORS[ct], label=ct) for ct in CTS]
fig.legend(handles=ct_legend, loc='lower center', ncol=6,
           fontsize=7, bbox_to_anchor=(0.5, -0.02))

# colorbar for direction
sm_red  = ScalarMappable(cmap='Reds',  norm=Normalize(0,1))
sm_blue = ScalarMappable(cmap='Blues', norm=Normalize(0,1))
cax1 = fig.add_axes([0.92, 0.55, 0.015, 0.3])
cax2 = fig.add_axes([0.92, 0.15, 0.015, 0.3])
fig.colorbar(sm_red,  cax=cax1, label='Increasing with age')
fig.colorbar(sm_blue, cax=cax2, label='Decreasing with age')

plt.tight_layout()
out = os.path.join(PLOT_DIR, 'circular_arrows_per_region.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ── PLOT 2: Global vs specific (grid: regions x 2) ───────────────────────────
print("Plotting Type 2: Global vs specific...")
fig, axes = plt.subplots(len(region_data), 2,
                          figsize=(12, len(region_data)*4))

global_threshold = 6

for idx, (region, df) in enumerate(region_data.items()):
    for col, (label, n_min, n_max) in enumerate([
        (f'Global (≥{global_threshold} regions)', global_threshold, 99),
        ('Region-specific (≤2 regions)', 0, 2)
    ]):
        ax = axes[idx, col]
        sig = df[df['age_qval'] < args.qval_thresh]

        # filter by cross-region count
        def in_range(row):
            key = (row['lr_pair'], row['source_ct'], row['target_ct'])
            count = lr_region_counts.get(key, 0)
            return n_min <= count <= n_max

        mask = sig.apply(in_range, axis=1)
        sub = sig[mask]

        if len(sub) == 0:
            ax.set_title(f'{region}\n{label}\n(no data)', fontsize=7)
            ax.axis('off')
            continue

        conns = get_connections(df, df['lr_pair'].isin(sub['lr_pair']) &
                                df['source_ct'].isin(sub['source_ct']) &
                                (df['age_qval'] < args.qval_thresh))
        active = sorted(set([c[0] for c in conns] + [c[1] for c in conns]))
        draw_circular_arrow_plot(ax, conns, active, f'{region}\n{label}')

fig.text(0.5, 1.005,
         f'Global (≥{global_threshold} regions) vs region-specific (≤2 regions) LR pairs\n'
         f'Arrow color: Red=increasing / Blue=decreasing with age',
         ha='center', fontsize=11, fontweight='bold')

ct_legend = [mpatches.Patch(facecolor=CT_COLORS[ct], label=ct) for ct in CTS]
fig.legend(handles=ct_legend, loc='lower center', ncol=6,
           fontsize=7, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout()
out = os.path.join(PLOT_DIR, 'circular_arrows_global_vs_specific.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ── PLOT 3: By functional category ───────────────────────────────────────────
print("Plotting Type 3: By functional category...")

top_cats = all_sig.merge(annot, on='lr_pair', how='left')
top_cats = top_cats['broad_category'].value_counts().head(6).index.tolist()

fig, axes = plt.subplots(len(region_data), len(top_cats),
                          figsize=(len(top_cats)*4, len(region_data)*4))

for idx, (region, df) in enumerate(region_data.items()):
    for j, cat in enumerate(top_cats):
        ax = axes[idx, j]
        sig = df[(df['age_qval'] < args.qval_thresh) &
                 (df['broad_category'] == cat)]
        if len(sig) == 0:
            ax.set_title(f'{region}\n{cat}\n(no data)', fontsize=6)
            ax.axis('off')
            continue
        conns = get_connections(df,
                                (df['age_qval'] < args.qval_thresh) &
                                (df['broad_category'] == cat))
        active = sorted(set([c[0] for c in conns] + [c[1] for c in conns]))
        title = f'{region}\n{cat}' if idx == 0 else region
        draw_circular_arrow_plot(ax, conns, active, title)

# column headers
for j, cat in enumerate(top_cats):
    fig.text((j + 0.5) / len(top_cats), 1.005, cat,
             ha='center', fontsize=9, fontweight='bold',
             color=CAT_COLORS.get(cat,'#333333'))

ct_legend = [mpatches.Patch(facecolor=CT_COLORS[ct], label=ct) for ct in CTS]
fig.legend(handles=ct_legend, loc='lower center', ncol=6,
           fontsize=7, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout()
out = os.path.join(PLOT_DIR, 'circular_arrows_by_category.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

print('\nAll chord/arrow plots saved.')
