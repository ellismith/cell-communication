#!/usr/bin/env python3
"""
plot_regression_region_celltype_heatmap_nanfilt.py
Same heatmaps as plot_regression_region_celltype_heatmap.py but using
only the regions covered by the nanfilt mashr runs, for direct comparison.
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
parser.add_argument('--qval_thresh', type=float, default=0.05)
parser.add_argument('--min_animals', type=int, default=10)
args = parser.parse_args()

BASE     = '/scratch/easmit31/cell_cell/results/within_region_analysis'
MASHR_DIR = '/scratch/easmit31/cell_cell/results/mashr'
PLOT_DIR = '/scratch/easmit31/cell_cell/results/regression_plots'
os.makedirs(PLOT_DIR, exist_ok=True)

REGIONS = ['ACC','CN','dlPFC','EC','HIP','IPP','lCb','M1','MB','mdTN','NAc']
CTS = ['Astrocyte','GABA','Glutamatergic','Microglia','Oligo','OPC',
       'Vascular','Basket','Cerebellar','Ependymal','Midbrain','MSN']

# get which regions were actually covered by nanfilt sender runs
def find_nanfilt(ct, mode):
    for nan_tag in ['0.55','0.54','0.52','0.50','0.48','0.46']:
        label = ct + '_' + mode + '_louvain_nanfilt' + nan_tag
        path  = os.path.join(MASHR_DIR, label, 'mashr_' + label + '_results.csv')
        if os.path.exists(path): return path
    return None

# collect which regions each CT had in nanfilt
nanfilt_regions = {}
for ct in CTS:
    path = find_nanfilt(ct, 'sender')
    if path is None: continue
    res = pd.read_csv(path)
    parts = res['condition'].str.split('|', expand=True)
    res['region'] = parts[2]
    nanfilt_regions[ct] = set(res['region'].unique())

# union of all nanfilt regions
all_nanfilt_regions = set()
for regions in nanfilt_regions.values():
    all_nanfilt_regions.update(regions)
nanfilt_region_list = [r for r in REGIONS if r in all_nanfilt_regions]
print(f'Regions covered by nanfilt runs: {nanfilt_region_list}')

# load regression data for those regions only
rows = []
for region in nanfilt_region_list:
    rlow = region.lower()
    path = f'{BASE}/regression_{region}/whole_{rlow}_age_sex_regression.csv'
    if not os.path.exists(path): continue

    print(f'  Loading {region}...')
    df = pd.read_csv(path, engine='c')
    df = df[df['n_animals'] >= args.min_animals]

    parts = df['interaction'].str.split('|', expand=True)
    df['source_ct'] = parts[0].str.rsplit('_', n=1).str[0]
    df['target_ct'] = parts[1].str.rsplit('_', n=1).str[0]
    df['lr_pair']   = parts[2] + '|' + parts[3]

    for (sender, receiver), grp in df.groupby(['source_ct','target_ct']):
        if sender not in CTS or receiver not in CTS: continue
        sig_grp = grp[grp['age_qval'] < args.qval_thresh]
        rows.append({
            'region':        region,
            'sender_ct':     sender,
            'receiver_ct':   receiver,
            'mean_beta_all': grp['age_coef'].mean(),
            'n_sig_lr':      sig_grp['lr_pair'].nunique(),
            'n_total_lr':    grp['lr_pair'].nunique(),
        })

df_agg = pd.DataFrame(rows)
df_agg['ct_pair'] = df_agg['sender_ct'] + u'\u2192' + df_agg['receiver_ct']

beta_all_mat = df_agg.pivot_table(index='ct_pair', columns='region',
                                   values='mean_beta_all', aggfunc='mean')
n_sig_mat    = df_agg.pivot_table(index='ct_pair', columns='region',
                                   values='n_sig_lr', aggfunc='sum').fillna(0)

beta_all_mat = beta_all_mat.reindex(columns=[r for r in nanfilt_region_list if r in beta_all_mat.columns])
n_sig_mat    = n_sig_mat.reindex(columns=[r for r in nanfilt_region_list if r in n_sig_mat.columns])

row_order = n_sig_mat.sum(axis=1).sort_values(ascending=False).index
beta_all_mat = beta_all_mat.reindex(row_order)
n_sig_mat    = n_sig_mat.reindex(row_order)

def make_heatmaps(beta_mat, n_sig_mat, suffix, title_suffix):
    # beta heatmap with asterisks
    vmax = np.nanpercentile(np.abs(beta_mat.values[~np.isnan(beta_mat.values)]), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(len(beta_mat.columns)*1.0+2,
                                     len(beta_mat)*0.28+2))
    im = ax.imshow(beta_mat.values, aspect='auto', cmap='RdBu_r', norm=norm)
    ax.set_xticks(range(len(beta_mat.columns)))
    ax.set_xticklabels(beta_mat.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(beta_mat.index)))
    ax.set_yticklabels(beta_mat.index, fontsize=7)
    plt.colorbar(im, ax=ax, label='mean age_coef (all LR pairs)')

    for i in range(len(beta_mat.index)):
        for j in range(len(beta_mat.columns)):
            region = beta_mat.columns[j]
            ct_pair = beta_mat.index[i]
            if region in n_sig_mat.columns and ct_pair in n_sig_mat.index:
                n = n_sig_mat.loc[ct_pair, region]
                if n >= 10:    marker = '***'
                elif n >= 5:   marker = '**'
                elif n >= 1:   marker = '*'
                else:          marker = ''
                if marker:
                    ax.text(j, i, marker, ha='center', va='center',
                            fontsize=5, color='black', fontweight='bold')

    ax.set_title(f'Mean age effect per cell type pair x region {title_suffix}\n'
                 f'All LR pairs; * ≥1 sig, ** ≥5 sig, *** ≥10 sig (qval<{args.qval_thresh})')
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'regression_region_celltype_allbeta_heatmap_{suffix}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

    # n sig heatmap
    fig, ax = plt.subplots(figsize=(len(n_sig_mat.columns)*1.0+2,
                                     len(n_sig_mat)*0.28+2))
    im = ax.imshow(np.log1p(n_sig_mat.values), aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(n_sig_mat.columns)))
    ax.set_xticklabels(n_sig_mat.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(n_sig_mat.index)))
    ax.set_yticklabels(n_sig_mat.index, fontsize=7)
    plt.colorbar(im, ax=ax, label='log1p(n significant LR pairs)')
    for i in range(len(n_sig_mat.index)):
        for j in range(len(n_sig_mat.columns)):
            val = int(n_sig_mat.values[i,j])
            if val > 0:
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=4, color='black' if val < n_sig_mat.values.max()*0.7 else 'white')
    ax.set_title(f'N significant LR pairs per cell type pair x region {title_suffix}\n'
                 f'(qval<{args.qval_thresh}, min_animals={args.min_animals})')
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'regression_region_celltype_nsig_heatmap_{suffix}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

# nanfilt-covered regions only
make_heatmaps(beta_all_mat, n_sig_mat, 'nanfilt_regions', '(nanfilt-covered regions only)')

# also make full 11-region version for comparison
print('\nLoading all regions for full comparison...')
rows_all = []
for region in REGIONS:
    rlow = region.lower()
    path = f'{BASE}/regression_{region}/whole_{rlow}_age_sex_regression.csv'
    if not os.path.exists(path): continue
    df = pd.read_csv(path, engine='c')
    df = df[df['n_animals'] >= args.min_animals]
    parts = df['interaction'].str.split('|', expand=True)
    df['source_ct'] = parts[0].str.rsplit('_', n=1).str[0]
    df['target_ct'] = parts[1].str.rsplit('_', n=1).str[0]
    df['lr_pair']   = parts[2] + '|' + parts[3]
    for (sender, receiver), grp in df.groupby(['source_ct','target_ct']):
        if sender not in CTS or receiver not in CTS: continue
        sig_grp = grp[grp['age_qval'] < args.qval_thresh]
        rows_all.append({
            'region': region, 'sender_ct': sender, 'receiver_ct': receiver,
            'mean_beta_all': grp['age_coef'].mean(),
            'n_sig_lr': sig_grp['lr_pair'].nunique(),
        })

df_all = pd.DataFrame(rows_all)
df_all['ct_pair'] = df_all['sender_ct'] + u'\u2192' + df_all['receiver_ct']
beta_all = df_all.pivot_table(index='ct_pair', columns='region',
                               values='mean_beta_all', aggfunc='mean')
n_sig_all = df_all.pivot_table(index='ct_pair', columns='region',
                                values='n_sig_lr', aggfunc='sum').fillna(0)
beta_all = beta_all.reindex(columns=[r for r in REGIONS if r in beta_all.columns])
n_sig_all = n_sig_all.reindex(columns=[r for r in REGIONS if r in n_sig_all.columns])
row_order2 = n_sig_all.sum(axis=1).sort_values(ascending=False).index
beta_all = beta_all.reindex(row_order2)
n_sig_all = n_sig_all.reindex(row_order2)
make_heatmaps(beta_all, n_sig_all, 'all_regions', '(all 11 regions)')

print('\nDone.')
