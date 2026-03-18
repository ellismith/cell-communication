#!/usr/bin/env python3
"""
plot_mashr_category_enrichment.py
Functional category enrichment analysis of mashr results.
For each sender cell type, tests whether each broad_category is enriched
among globally shared, intermediate, or condition-specific LR pairs
vs the background of all tested LR pairs.
Uses Fisher's exact test with FDR correction.
Outputs:
  - Heatmap of enrichment (-log10 FDR) per sender x category
  - Bar charts of category composition per sharing tier
  - CSV of all enrichment results
Usage:
    python plot_mashr_category_enrichment.py
    python plot_mashr_category_enrichment.py --sender_ct Astrocyte
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import argparse
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--sender_ct', type=str, default=None)
parser.add_argument('--lfsr_thresh', type=float, default=0.05)
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

annot = pd.read_csv(ANNOT)[['lr_pair','broad_category']].drop_duplicates()

CTS = ['Astrocyte','GABA','Glutamatergic','Microglia','Oligo','OPC',
       'Vascular','Basket','Cerebellar','Ependymal','Midbrain','MSN']
if args.sender_ct:
    CTS = [args.sender_ct]

all_enrich = []

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

    # assign sharing tier per LR pair
    def get_tier(lr):
        n = lr_counts.get(lr, 0)
        if n == 0:          return 'not_sig'
        if n == n_conds:    return 'global'
        if n > n_conds // 2: return 'broad'
        if n >= 6:          return 'intermediate'
        if n >= 2:          return 'specific_few'
        return 'specific_one'

    lr_annot = res[['lr_pair','broad_category']].drop_duplicates()
    lr_annot = lr_annot.merge(annot, on='lr_pair', how='left')
    lr_annot['broad_category'] = lr_annot['broad_category_y'].fillna(
                                  lr_annot['broad_category_x']).fillna('Unknown')
    lr_annot = lr_annot[['lr_pair','broad_category']].drop_duplicates('lr_pair')
    lr_annot['tier'] = lr_annot['lr_pair'].apply(get_tier)

    background = set(lr_annot['lr_pair'])
    categories = lr_annot['broad_category'].unique()
    tiers = ['global','broad','intermediate','specific_few','specific_one']

    for tier in tiers:
        tier_lrs = set(lr_annot[lr_annot['tier']==tier]['lr_pair'])
        if len(tier_lrs) == 0:
            continue
        for cat in categories:
            cat_lrs    = set(lr_annot[lr_annot['broad_category']==cat]['lr_pair'])
            a = len(tier_lrs & cat_lrs)
            b = len(tier_lrs - cat_lrs)
            c = len((background - tier_lrs) & cat_lrs)
            d = len(background - tier_lrs - cat_lrs)
            _, pval = fisher_exact([[a,b],[c,d]], alternative='greater')
            all_enrich.append({
                'sender': ct,
                'tier': tier,
                'category': cat,
                'n_tier': len(tier_lrs),
                'n_cat_in_tier': a,
                'pval': pval
            })

    print(f'  {ct}: done')

if not all_enrich:
    print('No enrichment results')
    exit()

df = pd.DataFrame(all_enrich)
_, df['qval'], _, _ = multipletests(df['pval'], method='fdr_bh')
df['neg_log10_q'] = -np.log10(df['qval'].clip(1e-300))

# save CSV
csv_out = os.path.join(PLOT_DIR, 'mashr_category_enrichment.csv')
df.to_csv(csv_out, index=False)
print(f'Saved: {csv_out}')

# ── HEATMAP: enrichment per sender x category (global tier only) ──────────────
for tier in ['global','intermediate','specific_one']:
    sub = df[df['tier']==tier]
    if len(sub) == 0:
        continue
    pivot = sub.pivot_table(index='category', columns='sender',
                            values='neg_log10_q', aggfunc='mean').fillna(0)
    pivot = pivot.loc[pivot.max(axis=1) > 1.3]  # keep categories with q<0.05 somewhere

    if len(pivot) == 0:
        continue

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)*0.8+2),
                                    max(4, len(pivot)*0.4+2)))
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=min(10, pivot.values.max()))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    plt.colorbar(im, ax=ax, label='-log10(FDR)')
    ax.set_title(f'Functional category enrichment — {tier} LR pairs\n(Fisher exact, FDR corrected)')
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'mashr_category_enrichment_{tier}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

# ── STACKED BAR: category composition per tier per sender ─────────────────────
for ct in CTS:
    sub = df[df['sender']==ct].copy()
    if len(sub) == 0:
        continue
    pivot = sub.pivot_table(index='tier', columns='category',
                            values='n_cat_in_tier', aggfunc='sum').fillna(0)
    pivot = pivot.reindex(['global','broad','intermediate','specific_few','specific_one'])
    pivot = pivot.dropna(how='all')

    # normalize to proportions
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(pivot_pct))
    for j, cat in enumerate(pivot_pct.columns):
        vals = pivot_pct[cat].values
        ax.bar(pivot_pct.index, vals, bottom=bottom,
               color=colors[j % len(colors)], label=cat)
        bottom += vals
    ax.set_ylabel('% of LR pairs')
    ax.set_title(f'{ct} sender — functional category composition by sharing tier')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'mashr_category_composition_{ct}_sender.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')
