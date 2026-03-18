#!/usr/bin/env python3
"""
plot_mashr_beta_heatmap_bytier.py
Per-sender heatmap of LR pairs x conditions colored by beta_posterior,
filtered to a specific sharing tier:
  - global:       sig in all conditions
  - broad:        sig in >50% conditions
  - intermediate: sig in 6-20 conditions
  - specific_few: sig in 2-5 conditions
  - specific_one: sig in exactly 1 condition
Cells where lfsr >= threshold are masked (shown as grey).
LR pairs labeled with functional broad_category.
Usage:
    python plot_mashr_beta_heatmap_bytier.py --tier intermediate
    python plot_mashr_beta_heatmap_bytier.py --sender_ct GABA --tier intermediate
    python plot_mashr_beta_heatmap_bytier.py --tier specific_one
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
parser.add_argument('--tier', type=str, default='intermediate',
                    choices=['global','broad','intermediate','specific_few','specific_one'])
parser.add_argument('--max_rows', type=int, default=80,
                    help='Max LR pairs to show (ranked by n conditions sig)')
args = parser.parse_args()

MASHR_DIR = '/scratch/easmit31/cell_cell/results/mashr'
ANNOT     = '/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv'
PLOT_DIR  = os.path.join(MASHR_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

annot = pd.read_csv(ANNOT)[['lr_pair','broad_category']].drop_duplicates('lr_pair')

CTS = ['Astrocyte','GABA','Glutamatergic','Microglia','Oligo','OPC',
       'Vascular','Basket','Cerebellar','Ependymal','Midbrain','MSN']
if args.sender_ct:
    CTS = [args.sender_ct]

def find_results(ct):
    for nan_tag in ['0.55','0.54','0.52','0.50','0.48','0.46']:
        label = f'{ct}_sender_louvain_nanfilt{nan_tag}'
        path  = os.path.join(MASHR_DIR, label, f'mashr_{label}_results.csv')
        if os.path.exists(path):
            return path, label
    return None, None

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
    n_conds = res['condition'].nunique()

    # select LR pairs for this tier
    if args.tier == 'global':
        tier_lrs = lr_counts[lr_counts == n_conds].index.tolist()
    elif args.tier == 'broad':
        tier_lrs = lr_counts[(lr_counts > n_conds//2) & (lr_counts < n_conds)].index.tolist()
    elif args.tier == 'intermediate':
        tier_lrs = lr_counts[(lr_counts >= 6) & (lr_counts <= 20)].index.tolist()
    elif args.tier == 'specific_few':
        tier_lrs = lr_counts[(lr_counts >= 2) & (lr_counts <= 5)].index.tolist()
    elif args.tier == 'specific_one':
        tier_lrs = lr_counts[lr_counts == 1].index.tolist()

    if len(tier_lrs) == 0:
        print(f'  {ct}: no LR pairs in tier {args.tier}')
        continue

    # limit rows
    tier_lrs = tier_lrs[:args.max_rows]
    print(f'  {ct}: {len(tier_lrs)} LR pairs in tier {args.tier}')

    sub = res[res['lr_pair'].isin(tier_lrs)]
    beta_mat = sub.pivot_table(index='lr_pair', columns='condition',
                                values='beta_posterior', aggfunc='mean')
    lfsr_mat = sub.pivot_table(index='lr_pair', columns='condition',
                                values='lfsr', aggfunc='mean')
    beta_mat = beta_mat.reindex(tier_lrs).dropna(how='all')
    lfsr_mat = lfsr_mat.reindex(beta_mat.index).reindex(columns=beta_mat.columns)

    lr_cats = res[['lr_pair','broad_category']].drop_duplicates('lr_pair').set_index('lr_pair')

    beta_plot = beta_mat.values.copy()
    mask = (lfsr_mat.values >= args.lfsr_thresh) | np.isnan(lfsr_mat.values)
    beta_plot[mask] = np.nan

    vmax = np.nanpercentile(np.abs(beta_plot[~np.isnan(beta_plot)]), 95) if np.any(~np.isnan(beta_plot)) else 0.01
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(10, len(beta_mat.columns)*0.12+2),
                                     max(6, len(beta_mat)*0.22+2)))
    im = ax.imshow(beta_plot, aspect='auto', cmap='RdBu_r', norm=norm)
    ax.set_xticks(range(len(beta_mat.columns)))
    ax.set_xticklabels(beta_mat.columns, rotation=90, fontsize=4)
    ax.set_yticks(range(len(beta_mat.index)))
    ylabels = [f'{lr} [{lr_cats.loc[lr,"broad_category"] if lr in lr_cats.index else "?"}]'
               for lr in beta_mat.index]
    ax.set_yticklabels(ylabels, fontsize=7)
    plt.colorbar(im, ax=ax, label='beta_posterior (age effect)')
    ax.set_title(f'{ct} sender — {args.tier} LR pairs (n={len(tier_lrs)})\nbeta_posterior (grey=lfsr≥{args.lfsr_thresh} or missing)')
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'mashr_beta_heatmap_{ct}_sender_{args.tier}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')
