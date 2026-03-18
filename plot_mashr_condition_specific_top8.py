#!/usr/bin/env python3
"""
plot_mashr_condition_specific.py
Table and bar chart of condition-specific hits (lfsr<thresh, sig in <=max_conds conditions).
Annotated with functional broad_category.
Usage:
    python plot_mashr_condition_specific.py
    python plot_mashr_condition_specific.py --sender_ct Astrocyte --max_conds 5
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
parser.add_argument('--max_conds', type=int, default=5)
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

annot = pd.read_csv(ANNOT)[['lr_pair','broad_category','classification']].drop_duplicates('lr_pair')

CTS = ['Astrocyte','GABA','Glutamatergic','Microglia','Oligo','OPC',
       'Vascular','Basket','Cerebellar','Ependymal','Midbrain','MSN']
if args.sender_ct:
    CTS = [args.sender_ct]

all_specific = []
for ct in CTS:
    path, label = find_results(ct)
    if path is None:
        print(f'  MISSING: {ct}')
        continue

    res = pd.read_csv(path)
    sig = res[res['lfsr'] < args.lfsr_thresh]
    lr_counts = sig['lr_pair'].value_counts()
    specific_lrs = lr_counts[lr_counts <= args.max_conds].index
    specific = sig[sig['lr_pair'].isin(specific_lrs)].copy()
    specific['sender'] = ct
    all_specific.append(specific)
    print(f'  {ct}: {len(specific_lrs)} condition-specific LR pairs')

if not all_specific:
    print('No condition-specific hits found')
    exit()

df = pd.concat(all_specific, ignore_index=True)
df = df.merge(annot, on='lr_pair', how='left')
df['broad_category'] = df['broad_category'].fillna('Unknown')
df = df.sort_values(['sender','lr_pair','lfsr'])

# save CSV
csv_out = os.path.join(PLOT_DIR, f'mashr_condition_specific_max{args.max_conds}.csv')
df[['sender','lr_pair','condition','beta_posterior','lfsr','beta_original',
    'broad_category','classification']].to_csv(csv_out, index=False)
print(f'Saved: {csv_out}')

# bar chart: count per sender colored by category
fig, ax = plt.subplots(figsize=(10, 4))
counts_by_cat = df.groupby(['sender','broad_category'])['lr_pair'].nunique().unstack(fill_value=0)
colors = plt.cm.tab20.colors
bottom = pd.Series(0, index=counts_by_cat.index)
for j, cat in enumerate(counts_by_cat.columns):
    ax.bar(counts_by_cat.index, counts_by_cat[cat], bottom=bottom,
           color=colors[j % len(colors)], label=cat)
    bottom += counts_by_cat[cat]
ax.set_ylabel(f'LR pairs sig in ≤{args.max_conds} conditions')
ax.set_title(f'Condition-specific LR pairs per sender (lfsr<{args.lfsr_thresh})\nColored by functional category')
plt.xticks(rotation=45, ha='right')
ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=7)
plt.tight_layout()
out = os.path.join(PLOT_DIR, f'mashr_condition_specific_counts.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
