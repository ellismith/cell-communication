#!/usr/bin/env python3
"""
plot_regression_per_region_dotplot.py
For each region, plots a dot plot showing age effects in cell-cell communication.
Rows = sender->receiver cell type pairs
Columns = top N LR pairs by significance in that region
Dot size = -log10(age_qval)
Dot color = age_coef direction (red=increasing, blue=decreasing)
Built directly from whole-region regression outputs.
Usage:
    python plot_regression_per_region_dotplot.py
    python plot_regression_per_region_dotplot.py --top_n 30 --qval_thresh 0.05
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--top_n', type=int, default=25,
                    help='Top N LR pairs to show per region')
parser.add_argument('--qval_thresh', type=float, default=0.05)
parser.add_argument('--min_animals', type=int, default=10)
parser.add_argument('--min_ct_pairs', type=int, default=2,
                    help='Min cell type pairs an LR pair must be sig in to show')
args = parser.parse_args()

BASE     = '/scratch/easmit31/cell_cell/results/within_region_analysis'
ANNOT    = '/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv'
PLOT_DIR = '/scratch/easmit31/cell_cell/results/regression_plots/per_region'
os.makedirs(PLOT_DIR, exist_ok=True)

REGIONS = ['ACC','CN','dlPFC','EC','HIP','IPP','lCb','M1','MB','mdTN','NAc']
CTS = ['Astrocyte','GABA','Glutamatergic','Microglia','Oligo','OPC',
       'Vascular','Basket','Cerebellar','Ependymal','Midbrain','MSN']

annot = pd.read_csv(ANNOT)[['lr_pair','broad_category']].drop_duplicates('lr_pair')

for region in REGIONS:
    rlow = region.lower()
    path = f'{BASE}/regression_{region}/whole_{rlow}_age_sex_regression.csv'
    if not os.path.exists(path):
        print(f'  MISSING: {region}')
        continue

    print(f'  Processing {region}...')
    df = pd.read_csv(path, engine='c')
    df = df[df['n_animals'] >= args.min_animals]

    parts = df['interaction'].str.split('|', expand=True)
    df['source_ct'] = parts[0].str.rsplit('_', n=1).str[0]
    df['target_ct'] = parts[1].str.rsplit('_', n=1).str[0]
    df['lr_pair']   = parts[2] + '|' + parts[3]
    df['ct_pair']   = df['source_ct'] + u'\u2192' + df['target_ct']

    # filter to known cell types
    df = df[df['source_ct'].isin(CTS) & df['target_ct'].isin(CTS)]

    if len(df) == 0:
        print(f'  {region}: no data after filtering')
        continue

    # get significant interactions
    sig = df[df['age_qval'] < args.qval_thresh]

    if len(sig) == 0:
        print(f'  {region}: no significant interactions')
        continue

    # select top N LR pairs by number of significant cell type pairs
    lr_ct_counts = sig.groupby('lr_pair')['ct_pair'].nunique()
    lr_ct_counts = lr_ct_counts[lr_ct_counts >= args.min_ct_pairs]
    if len(lr_ct_counts) == 0:
        # relax filter
        lr_ct_counts = sig.groupby('lr_pair')['ct_pair'].nunique()

    top_lrs = lr_ct_counts.nlargest(args.top_n).index.tolist()

    # for each lr_pair x ct_pair, get mean age_coef and min qval
    sub = df[df['lr_pair'].isin(top_lrs)]
    agg = sub.groupby(['lr_pair','ct_pair']).agg(
        mean_beta=('age_coef','mean'),
        min_qval=('age_qval','min'),
        n_louvain=('interaction','count')
    ).reset_index()

    # add annotations
    agg = agg.merge(annot, on='lr_pair', how='left')
    agg['broad_category'] = agg['broad_category'].fillna('Unknown')
    agg['neg_log10_q'] = -np.log10(agg['min_qval'].clip(1e-300))

    # get all ct_pairs present
    ct_pairs = sorted(agg['ct_pair'].unique())

    # order LR pairs by category then name
    lr_order = agg[agg['min_qval'] < args.qval_thresh].groupby('lr_pair')['neg_log10_q'].max()
    lr_order = lr_order.reindex(top_lrs).fillna(0).sort_values(ascending=False)
    top_lrs_ordered = lr_order.index.tolist()

    # add category label to lr pair
    lr_cat = agg[['lr_pair','broad_category']].drop_duplicates('lr_pair').set_index('lr_pair')
    ylabels = [f'{lr} [{lr_cat.loc[lr,"broad_category"] if lr in lr_cat.index else "?"}]'
               for lr in top_lrs_ordered]

    fig, ax = plt.subplots(figsize=(max(10, len(ct_pairs)*0.5+2),
                                     max(6, len(top_lrs_ordered)*0.3+2)))

    for i, lr in enumerate(top_lrs_ordered):
        for j, ct_pair in enumerate(ct_pairs):
            row = agg[(agg['lr_pair']==lr) & (agg['ct_pair']==ct_pair)]
            if len(row) == 0:
                continue
            row = row.iloc[0]

            # size by significance
            size = min(row['neg_log10_q'] * 15, 300)

            # color by direction, grey if not significant
            if row['min_qval'] < args.qval_thresh:
                color = '#d73027' if row['mean_beta'] > 0 else '#4575b4'
                alpha = 0.85
            else:
                color = '#cccccc'
                alpha = 0.3
                size = size * 0.3

            ax.scatter(j, i, s=size, c=color, alpha=alpha,
                      edgecolors='none', zorder=3)

    ax.set_xticks(range(len(ct_pairs)))
    ax.set_xticklabels(ct_pairs, rotation=90, fontsize=6)
    ax.set_yticks(range(len(top_lrs_ordered)))
    ax.set_yticklabels(ylabels, fontsize=6.5)
    ax.set_xlim(-0.5, len(ct_pairs)-0.5)
    ax.set_ylim(-0.5, len(top_lrs_ordered)-0.5)
    ax.grid(True, alpha=0.2)
    ax.invert_yaxis()

    # size legend
    for q in [0.05, 0.01, 0.001]:
        s = min(-np.log10(q) * 15, 300)
        ax.scatter([], [], s=s, c='grey', alpha=0.7, label=f'q={q}')
    ax.legend(title='q-value', fontsize=7, loc='lower right')

    # color legend
    from matplotlib.lines import Line2D
    color_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#d73027',
               markersize=8, label='Increasing with age'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#4575b4',
               markersize=8, label='Decreasing with age'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#cccccc',
               markersize=8, label='Not significant'),
    ]
    fig.legend(handles=color_elements, loc='upper right', fontsize=7,
               bbox_to_anchor=(1.0, 1.0))

    ax.set_title(f'{region} — top {len(top_lrs_ordered)} LR pairs by significance\n'
                 f'Size=-log10(min qval), Color=age effect direction\n'
                 f'(age_qval<{args.qval_thresh}, min_animals={args.min_animals})')

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'dotplot_{region}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')

print('\nDone.')
