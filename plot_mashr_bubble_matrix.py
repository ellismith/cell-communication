#!/usr/bin/env python3
"""
plot_mashr_bubble_matrix.py
Bubble matrix: rows=sender cell type, columns=receiver cell type.
Each bubble shows:
  - Color: mean beta_posterior direction (red=increasing, blue=decreasing with age)
  - Size: regional specificity (large=highly region-specific, small=globally shared)
  - Opacity: number of significant LR pairs (more sig = more opaque)
Uses the mashr_master_table.csv built by build_mashr_master_table.py.
Usage:
    python plot_mashr_bubble_matrix.py
    python plot_mashr_bubble_matrix.py --source sender_run
    python plot_mashr_bubble_matrix.py --category 'Glutamate signaling'
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default=None,
                    choices=['sender_run','receiver_run',None])
parser.add_argument('--category', type=str, default=None,
                    help='Filter to specific broad_category')
parser.add_argument('--lfsr_thresh', type=float, default=0.05)
parser.add_argument('--min_sig', type=int, default=1,
                    help='Min significant LR pairs to show bubble')
args = parser.parse_args()

MASHR_DIR = '/scratch/easmit31/cell_cell/results/mashr'
PLOT_DIR  = os.path.join(MASHR_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(MASHR_DIR, 'mashr_master_table.csv'))

# filter
if args.source:
    df = df[df['source'] == args.source]
if args.category:
    df = df[df['broad_category'] == args.category]

# keep only significant rows
sig = df[df['n_sig'] >= args.min_sig]

CTS = ['Astrocyte','GABA','Glutamatergic','Microglia','Oligo','OPC',
       'Vascular','Basket','Cerebellar','Ependymal','Midbrain','MSN']

# aggregate per sender x receiver
agg = sig.groupby(['sender_ct','receiver_ct']).agg(
    n_sig_lr=('lr_pair','nunique'),
    mean_beta=('mean_beta_sig','mean'),
    mean_pct_sig=('pct_sig','mean'),  # regional specificity proxy
).reset_index()

# regional specificity: high pct_sig = globally shared (low specificity)
# invert so large bubble = more region-specific
agg['specificity'] = 1 - agg['mean_pct_sig']

# normalize bubble size
max_n = agg['n_sig_lr'].max()
agg['bubble_size'] = (agg['n_sig_lr'] / max_n) * 800 + 50

fig, ax = plt.subplots(figsize=(13, 11))

for _, row in agg.iterrows():
    if row['sender_ct'] not in CTS or row['receiver_ct'] not in CTS:
        continue
    x = CTS.index(row['receiver_ct'])
    y = CTS.index(row['sender_ct'])

    # color by mean beta direction
    beta = row['mean_beta']
    if pd.isna(beta):
        color = '#cccccc'
    elif beta > 0:
        intensity = min(1.0, abs(beta) / 0.005)
        color = plt.cm.Reds(0.3 + 0.7 * intensity)
    else:
        intensity = min(1.0, abs(beta) / 0.005)
        color = plt.cm.Blues(0.3 + 0.7 * intensity)

    # alpha by regional specificity
    alpha = 0.4 + 0.6 * row['specificity']

    ax.scatter(x, y, s=row['bubble_size'], c=[color], alpha=alpha,
               edgecolors='grey', linewidths=0.5, zorder=3)

ax.set_xticks(range(len(CTS)))
ax.set_xticklabels(CTS, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(CTS)))
ax.set_yticklabels(CTS, fontsize=9)
ax.set_xlabel('Receiver cell type', fontsize=11)
ax.set_ylabel('Sender cell type', fontsize=11)
ax.grid(True, alpha=0.2)
ax.set_xlim(-0.5, len(CTS)-0.5)
ax.set_ylim(-0.5, len(CTS)-0.5)

# legends
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
size_legend = [
    ax.scatter([],[], s=50+800*f/max_n, c='grey', alpha=0.7, label=f'{int(f)} LR pairs')
    for f in [1, int(max_n*0.25), int(max_n*0.5), max_n]
]
l1 = ax.legend(handles=size_legend, title='N significant LR pairs',
               loc='upper right', fontsize=7, title_fontsize=8)
ax.add_artist(l1)

color_elements = [
    Patch(facecolor=plt.cm.Reds(0.7), label='Increasing with age'),
    Patch(facecolor=plt.cm.Blues(0.7), label='Decreasing with age'),
]
alpha_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='grey',
           markersize=10, alpha=0.4, label='Globally shared (low specificity)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='grey',
           markersize=10, alpha=1.0, label='Region-specific (high specificity)'),
]
ax.legend(handles=color_elements+alpha_elements,
          loc='lower right', fontsize=7, title='Effect direction / specificity',
          title_fontsize=8)

cat_tag = f'_{args.category.replace(" ","_")}' if args.category else ''
src_tag = f'_{args.source}' if args.source else ''
title = f'Age effects in cell-cell communication\nBubble size=n sig LR pairs, Color=direction, Opacity=region specificity'
if args.category:
    title += f'\nCategory: {args.category}'
ax.set_title(title, fontsize=10)

plt.tight_layout()
out = os.path.join(PLOT_DIR, f'mashr_bubble_matrix{src_tag}{cat_tag}.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
