#!/usr/bin/env python3
"""
plot_mashr_mixture_weights.py
Bar chart of mashr mixture weights per sender cell type.
Shows balance between null, equal_effects, and condition-specific components.
Usage:
    python plot_mashr_mixture_weights.py
    python plot_mashr_mixture_weights.py --sender_ct Astrocyte --top_n 15
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sender_ct', type=str, default=None)
parser.add_argument('--top_n', type=int, default=15)
args = parser.parse_args()

MASHR_DIR = '/scratch/easmit31/cell_cell/results/mashr'
PLOT_DIR  = os.path.join(MASHR_DIR, 'plots_top8perregion')
os.makedirs(PLOT_DIR, exist_ok=True)

def find_results(ct, mode='sender'):
    # try top8perregion first
    label = ct + '_' + mode + '_louvain_top8perregion'
    path  = os.path.join(MASHR_DIR, label, 'mashr_' + label + '_results.csv')
    if os.path.exists(path):
        return path, label
    # fall back to nanfilt
    for nan_tag in ['0.55','0.54','0.52','0.50','0.48','0.46']:
        label = ct + '_' + mode + '_louvain_nanfilt' + nan_tag
        path  = os.path.join(MASHR_DIR, label, 'mashr_' + label + '_results.csv')
        if os.path.exists(path):
            return path, label
    return None, None

try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
except ImportError:
    print("rpy2 not available -- cannot read RDS files")
    exit(1)

CTS = ['Astrocyte','GABA','Glutamatergic','Microglia','Oligo','OPC',
       'Vascular','Basket','Cerebellar','Ependymal','Midbrain','MSN']
if args.sender_ct:
    CTS = [args.sender_ct]

fig, axes = plt.subplots(len(CTS), 1, figsize=(12, len(CTS) * 2.5))
if len(CTS) == 1:
    axes = [axes]

for ax, ct in zip(axes, CTS):
    for nan_tag in ['0.55', '0.50']:
        label = f'{ct}_sender_louvain_nanfilt{nan_tag}'
        rds   = os.path.join(MASHR_DIR, label, f'mashr_{label}_model.rds')
        if os.path.exists(rds):
            break
    else:
        ax.set_title(f'{ct}: MISSING')
        continue

    ro.r(f'm <- readRDS("{rds}")')
    pi_vals  = np.array(ro.r('m$fitted_g$pi'))
    pi_names = list(ro.r('names(m$fitted_g$pi)'))

    # sort by weight descending, take top N
    order  = np.argsort(pi_vals)[::-1][:args.top_n]
    vals   = pi_vals[order]
    names  = [pi_names[i] for i in order]

    # color by type
    colors = []
    for n in names:
        if 'null' in n:           colors.append('#999999')
        elif 'equal_effects' in n: colors.append('#2ca02c')
        elif 'simple_het' in n:    colors.append('#ff7f0e')
        elif 'ED' in n:            colors.append('#9467bd')
        else:                      colors.append('#1f77b4')

    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_title(f'{ct} sender', fontsize=9)
    ax.set_xlabel('mixture weight', fontsize=7)

plt.suptitle('mashr mixture weights by sender cell type', fontsize=12, y=1.01)
plt.tight_layout()
out = os.path.join(PLOT_DIR, 'mashr_mixture_weights.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
