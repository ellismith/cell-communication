#!/usr/bin/env python3
"""
plot_mashr_per_region_dotplot_top8.py
For each region, plots a dot plot showing age effects from mashr top8perregion results.
Rows = sender->receiver cell type pairs (from Louvain combo conditions)
Columns = top N LR pairs by significance in that region
Dot size = number of significant Louvain combos in that region
Dot color = mean beta_posterior direction (red=increasing, blue=decreasing)
Grey = not significant (lfsr >= threshold)
Uses top8perregion mashr results (both sender and receiver runs).
Usage:
    python plot_mashr_per_region_dotplot_top8.py
    python plot_mashr_per_region_dotplot_top8.py --top_n 30 --lfsr_thresh 0.05
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--top_n', type=int, default=25)
parser.add_argument('--lfsr_thresh', type=float, default=0.05)
parser.add_argument('--mode', type=str, default='sender',
                    choices=['sender','receiver','both'])
args = parser.parse_args()

MASHR_DIR = '/scratch/easmit31/cell_cell/results/mashr'
ANNOT     = '/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv'
PLOT_DIR  = os.path.join(MASHR_DIR, 'plots_top8perregion', 'per_region')
os.makedirs(PLOT_DIR, exist_ok=True)

REGIONS = ['ACC','CN','dlPFC','EC','HIP','IPP','lCb','M1','MB','mdTN','NAc']
CTS = ['Astrocyte','GABA','Glutamatergic','Microglia','Oligo','OPC',
       'Vascular','Basket','Cerebellar','Ependymal','Midbrain','MSN']

annot = pd.read_csv(ANNOT)[['lr_pair','broad_category']].drop_duplicates('lr_pair')

def find_top8(ct, mode):
    label = ct + '_' + mode + '_louvain_top8perregion'
    path  = os.path.join(MASHR_DIR, label, 'mashr_' + label + '_results.csv')
    return path if os.path.exists(path) else None

# load all mashr results and parse region/cell types from conditions
print("Loading mashr results...")
all_dfs = []
modes = ['sender','receiver'] if args.mode == 'both' else [args.mode]

for mode in modes:
    for ct in CTS:
        path = find_top8(ct, mode)
        if path is None: continue
        res = pd.read_csv(path)
        parts = res['condition'].str.split('|', expand=True)
        res['source_lou'] = parts[0]
        res['target_lou'] = parts[1]
        res['region']     = parts[2]
        res['source_ct']  = res['source_lou'].str.rsplit('_', n=1).str[0]
        res['target_ct']  = res['target_lou'].str.rsplit('_', n=1).str[0]
        res['ct_pair']    = res['source_ct'] + u'\u2192' + res['target_ct']
        res['run_mode']   = mode
        all_dfs.append(res)
        print(f'  loaded {ct} {mode}')

df = pd.concat(all_dfs, ignore_index=True)
df = df.merge(annot, on='lr_pair', how='left')
df['broad_category'] = df['broad_category'].fillna('Unknown')

print(f'Total rows loaded: {len(df):,}')

# process each region
for region in REGIONS:
    sub = df[df['region'] == region]
    if len(sub) == 0:
        print(f'  {region}: no data')
        continue

    sig = sub[sub['lfsr'] < args.lfsr_thresh]
    if len(sig) == 0:
        print(f'  {region}: no significant interactions')
        continue

    # select top N LR pairs by number of significant ct_pairs
    lr_ct_counts = sig.groupby('lr_pair')['ct_pair'].nunique()
    top_lrs = lr_ct_counts.nlargest(args.top_n).index.tolist()

    # for each lr_pair x ct_pair, aggregate across Louvain combos
    sub_top = sub[sub['lr_pair'].isin(top_lrs)]
    agg = sub_top.groupby(['lr_pair','ct_pair']).agg(
        mean_beta=('beta_posterior','mean'),
        min_lfsr=('lfsr','min'),
        n_sig_louvain=('lfsr', lambda x: (x < args.lfsr_thresh).sum()),
        n_total_louvain=('lfsr','count')
    ).reset_index()

    # add annotations
    lr_cat = df[['lr_pair','broad_category']].drop_duplicates('lr_pair').set_index('lr_pair')

    # get ct_pairs present
    ct_pairs = sorted(agg[agg['min_lfsr'] < args.lfsr_thresh]['ct_pair'].unique())
    if len(ct_pairs) == 0:
        ct_pairs = sorted(agg['ct_pair'].unique())

    # order LR pairs by max significance
    lr_order = agg[agg['min_lfsr'] < args.lfsr_thresh].groupby('lr_pair')['n_sig_louvain'].max()
    lr_order = lr_order.reindex(top_lrs).fillna(0).sort_values(ascending=False)
    top_lrs_ordered = lr_order.index.tolist()

    ylabels = [f'{lr} [{lr_cat.loc[lr,"broad_category"] if lr in lr_cat.index else "?"}]'
               for lr in top_lrs_ordered]

    fig, ax = plt.subplots(figsize=(max(10, len(ct_pairs)*0.45+2),
                                     max(6, len(top_lrs_ordered)*0.3+2)))

    max_n_sig = agg['n_sig_louvain'].max()

    for i, lr in enumerate(top_lrs_ordered):
        for j, ct_pair in enumerate(ct_pairs):
            row = agg[(agg['lr_pair']==lr) & (agg['ct_pair']==ct_pair)]
            if len(row) == 0:
                continue
            row = row.iloc[0]

            n_sig = row['n_sig_louvain']
            if n_sig == 0:
                continue

            # size by n significant Louvain combos
            size = max(20, (n_sig / max(max_n_sig, 1)) * 300)

            # color by direction
            if row['min_lfsr'] < args.lfsr_thresh:
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
    for n in [1, 3, 5, max_n_sig]:
        s = max(20, (n / max(max_n_sig, 1)) * 300)
        ax.scatter([], [], s=s, c='grey', alpha=0.7, label=f'n={int(n)}')
    ax.legend(title='N sig Louvain combos', fontsize=7, loc='lower right')

    from matplotlib.lines import Line2D
    color_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#d73027',
               markersize=8, label='Increasing with age'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#4575b4',
               markersize=8, label='Decreasing with age'),
    ]
    fig.legend(handles=color_elements, loc='upper right', fontsize=7,
               bbox_to_anchor=(1.0, 1.0))

    ax.set_title(f'{region} — top {len(top_lrs_ordered)} LR pairs (mashr top8perregion)\n'
                 f'Size=n sig Louvain combos, Color=direction (lfsr<{args.lfsr_thresh})\n'
                 f'Mode: {args.mode}')

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'mashr_dotplot_{region}_{args.mode}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')

print('\nDone.')
