#!/usr/bin/env python3
"""
plot_mashr_summary_heatmap.py
Cross-sender summary heatmap: rows=LR pairs, columns=sender cell types,
value=number of conditions significant at lfsr<threshold.
LR pairs annotated with broad_category on y-axis.
Usage:
    python plot_mashr_summary_heatmap.py
    python plot_mashr_summary_heatmap.py --min_senders 2 --top_n 60
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lfsr_thresh', type=float, default=0.05)
parser.add_argument('--min_senders', type=int, default=2)
parser.add_argument('--top_n', type=int, default=60)
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

ct_counts = {}
for ct in CTS:
    path, label = find_results(ct)
    if path is None:
        print(f'  MISSING: {ct}')
        continue
    res = pd.read_csv(path)
    sig = res[res['lfsr'] < args.lfsr_thresh]
    ct_counts[ct] = sig.groupby('lr_pair').size()
    print(f'  loaded {ct}')

if not ct_counts:
    print('No results found')
    exit()

mat = pd.DataFrame(ct_counts).fillna(0).astype(int)
n_senders_sig = (mat > 0).sum(axis=1)
mat = mat[n_senders_sig >= args.min_senders]
mat['total'] = mat.sum(axis=1)
mat = mat.nlargest(args.top_n, 'total').drop(columns='total')

# join annotations for y-axis labels
mat = mat.merge(annot, left_index=True, right_on='lr_pair', how='left')
mat['broad_category'] = mat['broad_category'].fillna('Unknown')
mat = mat.set_index('lr_pair')
ylabels = [f'{lr} [{mat.loc[lr,"broad_category"]}]' for lr in mat.index]
mat = mat.drop(columns='broad_category')

fig, ax = plt.subplots(figsize=(max(8, len(mat.columns)*0.8+2),
                                 max(8, len(mat)*0.22+2)))
im = ax.imshow(mat.values, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(mat.columns)))
ax.set_xticklabels(mat.columns, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(mat.index)))
ax.set_yticklabels(ylabels, fontsize=6)
plt.colorbar(im, ax=ax, label='N conditions significant')
ax.set_title(f'Cross-sender LR pair significance\n(lfsr<{args.lfsr_thresh}, top {args.top_n} LR pairs, ≥{args.min_senders} senders)')
plt.tight_layout()
out = os.path.join(MASHR_DIR, 'plots', 'mashr_summary_heatmap.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

csv_out = os.path.join(PLOT_DIR, 'mashr_summary_heatmap.csv')
mat.to_csv(csv_out)
print(f'Saved: {csv_out}')
