#!/usr/bin/env python3
"""
plot_mashr_beta_heatmap.py
Per-sender heatmap of LR pairs x conditions colored by beta_posterior.
Cells where lfsr >= threshold are masked (shown as grey).
LR pairs labeled with functional broad_category.
Usage:
    python plot_mashr_beta_heatmap.py
    python plot_mashr_beta_heatmap.py --sender_ct Astrocyte --top_n 50
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import TwoSlopeNorm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sender_ct', type=str, default=None)
parser.add_argument('--lfsr_thresh', type=float, default=0.05)
parser.add_argument('--top_n', type=int, default=50)
args = parser.parse_args()

MASHR_DIR = '/scratch/easmit31/cell_cell/results/mashr'
ANNOT     = '/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv'
PLOT_DIR  = os.path.join(MASHR_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

def find_results(ct):
    for nan_tag in ['0.55','0.54','0.52','0.50','0.48','0.46']:
        label = f'{ct}_sender_louvain_nanfilt{nan_tag}'
        path  = os.path.join(MASHR_DIR, label, f'mashr_{label}_results.csv')
        if os.path.exists(path):
            return path, label
    return None, None

annot = pd.read_csv(ANNOT)[['lr_pair','broad_category']].drop_duplicates('lr_pair')

CTS = ['Astrocyte','GABA','Glutamatergic','Microglia','Oligo','OPC',
       'Vascular','Basket','Cerebellar','Ependymal','Midbrain','MSN']
if args.sender_ct:
    CTS = [args.sender_ct]

for ct in CTS:
    path, label = find_results(ct)
    if path is None:
        print(f'  MISSING: {ct}')
        continue

    res = pd.read_csv(path)
    res = res.merge(annot, on='lr_pair', how='left')
    res['broad_category'] = res['broad_category'].fillna('Unknown')

    sig = res[res['lfsr'] < args.lfsr_thresh]
    lr_counts = sig['lr_pair'].value_counts()
    if len(lr_counts) == 0:
        print(f'  {ct}: no significant LR pairs')
        continue

    top_lrs = lr_counts.head(args.top_n).index.tolist()
    sub = res[res['lr_pair'].isin(top_lrs)]

    beta_mat = sub.pivot_table(index='lr_pair', columns='condition',
                                values='beta_posterior', aggfunc='mean')
    lfsr_mat = sub.pivot_table(index='lr_pair', columns='condition',
                                values='lfsr', aggfunc='mean')
    beta_mat = beta_mat.loc[[lr for lr in top_lrs if lr in beta_mat.index]]
    lfsr_mat = lfsr_mat.reindex(beta_mat.index).reindex(columns=beta_mat.columns)

    # get category label for each LR pair
    lr_cats = res[['lr_pair','broad_category']].drop_duplicates('lr_pair').set_index('lr_pair')

    beta_plot = beta_mat.values.copy()
    mask = lfsr_mat.values >= args.lfsr_thresh
    beta_plot[mask] = np.nan

    vmax = np.nanpercentile(np.abs(beta_plot), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(10, len(beta_mat.columns)*0.15+2),
                                     max(6, len(beta_mat)*0.2+2)))
    im = ax.imshow(beta_plot, aspect='auto', cmap='RdBu_r', norm=norm)
    ax.set_xticks(range(len(beta_mat.columns)))
    ax.set_xticklabels(beta_mat.columns, rotation=90, fontsize=5)
    ax.set_yticks(range(len(beta_mat.index)))

    # label rows with lr_pair + category
    ylabels = [f'{lr} [{lr_cats.loc[lr,"broad_category"] if lr in lr_cats.index else "?"}]'
               for lr in beta_mat.index]
    ax.set_yticklabels(ylabels, fontsize=6)
    plt.colorbar(im, ax=ax, label='beta_posterior (age effect)')
    ax.set_title(f'{ct} sender — top {args.top_n} LR pairs\nbeta_posterior (grey=lfsr≥{args.lfsr_thresh})')
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'mashr_beta_heatmap_{ct}_sender.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')
