#!/usr/bin/env python3
"""
plot_mashr_covariance.py
Heatmap of the dominant learned mashr covariance matrix per sender.
Shows which conditions (louvain combos x regions) co-vary in age effects.
Usage:
    python plot_mashr_covariance.py
    python plot_mashr_covariance.py --sender_ct Astrocyte
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import TwoSlopeNorm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sender_ct', type=str, default=None)
args = parser.parse_args()

MASHR_DIR = '/scratch/easmit31/cell_cell/results/mashr'
PLOT_DIR  = os.path.join(MASHR_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

def find_results(ct):
    for nan_tag in ['0.55','0.54','0.52','0.50','0.48','0.46']:
        label = f'{ct}_sender_louvain_nanfilt{nan_tag}'
        path  = os.path.join(MASHR_DIR, label, f'mashr_{label}_results.csv')
        if os.path.exists(path):
            return path, label
    return None, None

try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
except ImportError:
    print("rpy2 not available -- cannot read RDS files directly")
    print("Run this script in mashr_env or convert RDS to CSV first")
    exit(1)

CTS = ['Astrocyte','GABA','Glutamatergic','Microglia','Oligo','OPC',
       'Vascular','Basket','Cerebellar','Ependymal','Midbrain','MSN']
if args.sender_ct:
    CTS = [args.sender_ct]

for ct in CTS:
    _, label = find_results(ct)
    if label is None:
        print(f'  MISSING: {ct}')
        continue
    rds = os.path.join(MASHR_DIR, label, f'mashr_{label}_model.rds')
    if not os.path.exists(rds):
        print(f'  MISSING RDS: {ct}')
        continue

    # load model and extract dominant covariance
    ro.r(f'm <- readRDS("{rds}")')
    ro.r('pi <- m$fitted_g$pi')
    ro.r('Ulist <- m$fitted_g$Ulist')
    ro.r('top_idx <- which.max(pi[-1]) + 1')  # exclude null
    ro.r('top_U <- Ulist[[top_idx]]')
    ro.r('top_name <- names(Ulist)[top_idx]')

    U = np.array(ro.r('top_U'))
    top_name = str(ro.r('top_name')[0])
    cond_names = list(ro.r('colnames(m$result$PosteriorMean)'))

    vmax = np.percentile(np.abs(U), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(8, len(cond_names)*0.15+2),
                                     max(8, len(cond_names)*0.15+2)))
    im = ax.imshow(U, aspect='auto', cmap='RdBu_r', norm=norm)
    ax.set_xticks(range(len(cond_names)))
    ax.set_xticklabels(cond_names, rotation=90, fontsize=4)
    ax.set_yticks(range(len(cond_names)))
    ax.set_yticklabels(cond_names, fontsize=4)
    plt.colorbar(im, ax=ax)
    ax.set_title(f'{ct} sender — dominant covariance: {top_name}')
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'mashr_covariance_{ct}_sender.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')
