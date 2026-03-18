#!/usr/bin/env python3
"""
plot_mashr_regional_dotplot.py
Dot plot: x=region, y=LR pair (annotated with broad_category).
Size = number of significant Louvain combos in that region.
Color = mean beta_posterior direction (red=increase, blue=decrease with age).
Usage:
    python plot_mashr_regional_dotplot.py
    python plot_mashr_regional_dotplot.py --sender_ct Astrocyte --top_n 30
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sender_ct', type=str, default=None)
parser.add_argument('--lfsr_thresh', type=float, default=0.05)
parser.add_argument('--top_n', type=int, default=30)
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
REGIONS = ['ACC','CN','dlPFC','EC','HIP','IPP','lCb','M1','MB','mdTN','NAc']

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

    sig = res[res['lfsr'] < args.lfsr_thresh].copy()
    if len(sig) == 0:
        print(f'  {ct}: no significant hits')
        continue

    sig['region'] = sig['condition'].str.split('|').str[-1]
    top_lrs = sig['lr_pair'].value_counts().head(args.top_n).index.tolist()
    sig = sig[sig['lr_pair'].isin(top_lrs)]

    agg = sig.groupby(['lr_pair','region']).agg(
        n_sig=('lfsr','count'),
        mean_beta=('beta_posterior','mean')
    ).reset_index()

    n_mat    = agg.pivot(index='lr_pair', columns='region', values='n_sig').reindex(columns=REGIONS).fillna(0)
    beta_mat = agg.pivot(index='lr_pair', columns='region', values='mean_beta').reindex(columns=REGIONS)
    n_mat    = n_mat.loc[top_lrs]
    beta_mat = beta_mat.reindex(n_mat.index)

    # y-axis labels with category
    lr_cats = res[['lr_pair','broad_category']].drop_duplicates('lr_pair').set_index('lr_pair')
    ylabels = [f'{lr} [{lr_cats.loc[lr,"broad_category"] if lr in lr_cats.index else "?"}]'
               for lr in n_mat.index]

    fig, ax = plt.subplots(figsize=(len(REGIONS)*0.7+2, len(top_lrs)*0.35+2))
    for i, lr in enumerate(n_mat.index):
        for j, reg in enumerate(REGIONS):
            n    = n_mat.loc[lr, reg]
            beta = beta_mat.loc[lr, reg]
            if n > 0 and not pd.isna(beta):
                color = '#d73027' if beta > 0 else '#4575b4'
                ax.scatter(j, i, s=n*20, color=color, alpha=0.8, zorder=3)

    ax.set_xticks(range(len(REGIONS)))
    ax.set_xticklabels(REGIONS, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(top_lrs)))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlim(-0.5, len(REGIONS)-0.5)
    ax.set_ylim(-0.5, len(top_lrs)-0.5)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    ax.set_title(f'{ct} sender — regional dot plot\nSize=n sig Louvain combos, Red=↑ Blue=↓ with age')
    for s in [1, 5, 10]:
        ax.scatter([], [], s=s*20, color='grey', alpha=0.8, label=f'n={s}')
    ax.legend(title='n sig combos', fontsize=7, loc='lower right')
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'mashr_regional_dotplot_{ct}_sender.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')
