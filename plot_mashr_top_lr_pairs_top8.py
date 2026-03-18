#!/usr/bin/env python3
"""
plot_mashr_top_lr_pairs.py
Bar chart of top N most significant LR pairs per sender cell type,
ranked by number of conditions significant at lfsr < threshold.
Bars annotated with functional broad_category.
Usage:
    python plot_mashr_top_lr_pairs.py
    python plot_mashr_top_lr_pairs.py --sender_ct Astrocyte --top_n 20
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sender_ct', type=str, default=None)
parser.add_argument('--lfsr_thresh', type=float, default=0.05)
parser.add_argument('--top_n', type=int, default=20)
args = parser.parse_args()

MASHR_DIR = '/scratch/easmit31/cell_cell/results/mashr'
ANNOT     = '/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv'
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
    sig = res[res['lfsr'] < args.lfsr_thresh]
    lr_counts = sig['lr_pair'].value_counts().head(args.top_n)
    mean_beta = sig.groupby('lr_pair')['beta_posterior'].mean()

    if len(lr_counts) == 0:
        print(f'  {ct}: no significant LR pairs')
        continue

    # join annotations
    lr_df = pd.DataFrame({'lr_pair': lr_counts.index, 'n_sig': lr_counts.values})
    lr_df = lr_df.merge(annot, on='lr_pair', how='left')
    lr_df['broad_category'] = lr_df['broad_category'].fillna('Unknown')
    lr_df['mean_beta'] = lr_df['lr_pair'].map(mean_beta)

    # color by broad_category
    categories = lr_df['broad_category'].unique()
    cmap = plt.cm.tab20.colors
    cat_colors = {c: cmap[i % len(cmap)] for i, c in enumerate(categories)}
    colors = [cat_colors[c] for c in lr_df['broad_category']]

    fig, ax = plt.subplots(figsize=(10, max(4, len(lr_df) * 0.35 + 1)))
    bars = ax.barh(range(len(lr_df)), lr_df['n_sig'], color=colors)
    ax.set_yticks(range(len(lr_df)))
    ax.set_yticklabels(lr_df['lr_pair'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Number of conditions significant')
    ax.set_title(f'{ct} sender — top {args.top_n} LR pairs (lfsr<{args.lfsr_thresh})\nColored by functional category')
    ax.axvline(res['condition'].nunique(), color='gray', linestyle='--', alpha=0.5, label='n conditions')

    # legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cat_colors[c], label=c) for c in categories]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1),
              loc='upper left', fontsize=7)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'mashr_top_lr_pairs_{ct}_sender.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')
