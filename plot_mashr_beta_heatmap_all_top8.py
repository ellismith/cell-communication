#!/usr/bin/env python3
"""
plot_mashr_beta_heatmap_all.py
Per-sender heatmap of ALL LR pairs x conditions colored by beta_posterior.
Unlike plot_mashr_beta_heatmap.py which shows top N by significance,
this shows all LR pairs including non-significant ones.
Cells where lfsr >= threshold are masked (shown as grey).
Two versions saved:
  - _all: full beta scale (95th percentile)
  - _all_capped: capped at fixed vmax for cleaner visualization
Usage:
    python plot_mashr_beta_heatmap_all.py
    python plot_mashr_beta_heatmap_all.py --sender_ct GABA
    python plot_mashr_beta_heatmap_all.py --sender_ct GABA --vmax 0.003
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
parser.add_argument('--vmax', type=float, default=None,
                    help='Fixed colorscale cap. If not set, uses 95th percentile.')
parser.add_argument('--tier', type=str, default=None,
                    choices=['global','broad','intermediate','specific_few','specific_one',None])
args = parser.parse_args()

MASHR_DIR = '/scratch/easmit31/cell_cell/results/mashr'
ANNOT     = '/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv'
PLOT_DIR  = os.path.join(MASHR_DIR, 'plots_top8perregion')
os.makedirs(PLOT_DIR, exist_ok=True)

annot = pd.read_csv(ANNOT)[['lr_pair','broad_category']].drop_duplicates('lr_pair')

CTS = ['Astrocyte','GABA','Glutamatergic','Microglia','Oligo','OPC',
       'Vascular','Basket','Cerebellar','Ependymal','Midbrain','MSN']
if args.sender_ct:
    CTS = [args.sender_ct]

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

def make_heatmap(beta_plot, lfsr_mat, beta_mat, ylabels, title, out, vmax=None):
    beta_masked = beta_plot.copy()
    mask = (lfsr_mat.values >= args.lfsr_thresh) | np.isnan(lfsr_mat.values)
    beta_masked[mask] = np.nan

    if vmax is None:
        vals = beta_masked[~np.isnan(beta_masked)]
        vmax = np.percentile(np.abs(vals), 95) if len(vals) > 0 else 0.01

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(max(10, len(beta_mat.columns)*0.12+2),
                                     max(6, len(beta_mat)*0.18+2)))
    im = ax.imshow(beta_masked, aspect='auto', cmap='RdBu_r', norm=norm)
    ax.set_xticks(range(len(beta_mat.columns)))
    ax.set_xticklabels(beta_mat.columns, rotation=90, fontsize=4)
    ax.set_yticks(range(len(beta_mat.index)))
    ax.set_yticklabels(ylabels, fontsize=6)
    plt.colorbar(im, ax=ax, label='beta_posterior (age effect)')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')

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

    # select LR pairs by tier if specified, otherwise all
    if args.tier is not None:
        if args.tier == 'global':
            lrs = lr_counts[lr_counts == n_conds].index.tolist()
        elif args.tier == 'broad':
            lrs = lr_counts[(lr_counts > n_conds//2) & (lr_counts < n_conds)].index.tolist()
        elif args.tier == 'intermediate':
            lrs = lr_counts[(lr_counts >= 6) & (lr_counts <= 20)].index.tolist()
        elif args.tier == 'specific_few':
            lrs = lr_counts[(lr_counts >= 2) & (lr_counts <= 5)].index.tolist()
        elif args.tier == 'specific_one':
            lrs = lr_counts[lr_counts == 1].index.tolist()
        tier_tag = f'_{args.tier}'
    else:
        lrs = res['lr_pair'].unique().tolist()
        tier_tag = '_all'

    if len(lrs) == 0:
        print(f'  {ct}: no LR pairs for tier {args.tier}')
        continue

    print(f'  {ct}: {len(lrs)} LR pairs')

    sub = res[res['lr_pair'].isin(lrs)]
    beta_mat = sub.pivot_table(index='lr_pair', columns='condition',
                                values='beta_posterior', aggfunc='mean')
    lfsr_mat = sub.pivot_table(index='lr_pair', columns='condition',
                                values='lfsr', aggfunc='mean')
    beta_mat = beta_mat.reindex(lrs).dropna(how='all')
    lfsr_mat = lfsr_mat.reindex(beta_mat.index).reindex(columns=beta_mat.columns)

    lr_cats = res[['lr_pair','broad_category']].drop_duplicates('lr_pair').set_index('lr_pair')
    ylabels = [f'{lr} [{lr_cats.loc[lr,"broad_category"] if lr in lr_cats.index else "?"}]'
               for lr in beta_mat.index]

    beta_plot = beta_mat.values.copy()

    # version 1: 95th percentile scale
    out1 = os.path.join(PLOT_DIR, f'mashr_beta_heatmap_{ct}_sender{tier_tag}.png')
    make_heatmap(beta_plot, lfsr_mat, beta_mat, ylabels,
                 f'{ct} sender — {len(lrs)} LR pairs{tier_tag}\nbeta_posterior (grey=lfsr≥{args.lfsr_thresh})',
                 out1, vmax=None)

    # version 2: capped scale
    cap = args.vmax if args.vmax else 0.003
    out2 = os.path.join(PLOT_DIR, f'mashr_beta_heatmap_{ct}_sender{tier_tag}_capped{cap}.png')
    make_heatmap(beta_plot, lfsr_mat, beta_mat, ylabels,
                 f'{ct} sender — {len(lrs)} LR pairs{tier_tag}\nbeta_posterior capped at ±{cap} (grey=lfsr≥{args.lfsr_thresh})',
                 out2, vmax=cap)
