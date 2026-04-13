#!/usr/bin/env python3
"""
plot_regression_region_celltype_heatmap.py
Heatmap of mean age_coef per sender->receiver cell type pair per region.
All LR pairs shown (not just significant), with asterisks marking significance.
Built directly from whole-region regression outputs.
Usage:
    python plot_regression_region_celltype_heatmap.py
    python plot_regression_region_celltype_heatmap.py --qval_thresh 0.05 --min_animals 10
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
PLOT_DIR = '/scratch/easmit31/cell_cell/results/regression_plots'
os.makedirs(PLOT_DIR, exist_ok=True)

REGIONS = ['ACC','CN','dlPFC','EC','HIP','IPP','lCb','M1','MB','mdTN','NAc']
CTS = ['Astrocyte','GABA','Glutamatergic','Microglia','Oligo','OPC',
       'Vascular','Basket','Cerebellar','Ependymal','Midbrain','MSN']

rows = []
for region in REGIONS:
    rlow = region.lower()
    path = f'{BASE}/regression_{region}/whole_{rlow}_age_sex_regression.csv'
    if not os.path.exists(path):
        print(f'  MISSING: {region}')
        continue

    print(f'  Loading {region}...')
    df = pd.read_csv(path, engine='c')
    df = df[df['n_animals'] >= args.min_animals]

    parts = df['interaction'].str.split('|', expand=True)
    df['source_ct'] = parts[0].str.rsplit('_', n=1).str[0]
    df['target_ct'] = parts[1].str.rsplit('_', n=1).str[0]
    df['lr_pair']   = parts[2] + '|' + parts[3]

    for (sender, receiver), grp in df.groupby(['source_ct','target_ct']):
        if sender not in CTS or receiver not in CTS:
            continue
        sig_grp = grp[grp['age_qval'] < args.qval_thresh]
        rows.append({
            'region':       region,
            'sender_ct':    sender,
            'receiver_ct':  receiver,
            'mean_beta_all': grp['age_coef'].mean(),
            'mean_beta_sig': sig_grp['age_coef'].mean() if len(sig_grp) > 0 else np.nan,
            'n_sig_lr':     sig_grp['lr_pair'].nunique(),
            'n_total_lr':   grp['lr_pair'].nunique(),
            'pct_sig':      len(sig_grp) / len(grp) if len(grp) > 0 else 0,
        })

df_agg = pd.DataFrame(rows)
df_agg['ct_pair'] = df_agg['sender_ct'] + u'\u2192' + df_agg['receiver_ct']

# pivot
beta_all_mat = df_agg.pivot_table(index='ct_pair', columns='region',
                                   values='mean_beta_all', aggfunc='mean')
n_sig_mat    = df_agg.pivot_table(index='ct_pair', columns='region',
                                   values='n_sig_lr', aggfunc='sum').fillna(0)

# reorder columns
beta_all_mat = beta_all_mat.reindex(columns=[r for r in REGIONS if r in beta_all_mat.columns])
n_sig_mat    = n_sig_mat.reindex(columns=[r for r in REGIONS if r in n_sig_mat.columns])

# sort rows by total sig LR pairs
row_order = n_sig_mat.sum(axis=1).sort_values(ascending=False).index
beta_all_mat = beta_all_mat.reindex(row_order)
n_sig_mat    = n_sig_mat.reindex(row_order)

# save CSV
df_agg.to_csv(os.path.join(PLOT_DIR, 'regression_region_celltype_all.csv'), index=False)

# ── PLOT: all betas with asterisks on significant ─────────────────────────────
vmax = np.nanpercentile(np.abs(beta_all_mat.values[~np.isnan(beta_all_mat.values)]), 95)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

fig, ax = plt.subplots(figsize=(len(beta_all_mat.columns)*1.0+2,
                                 len(beta_all_mat)*0.28+2))
im = ax.imshow(beta_all_mat.values, aspect='auto', cmap='RdBu_r', norm=norm)
ax.set_xticks(range(len(beta_all_mat.columns)))
ax.set_xticklabels(beta_all_mat.columns, rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(len(beta_all_mat.index)))
ax.set_yticklabels(beta_all_mat.index, fontsize=7)
plt.colorbar(im, ax=ax, label='mean age_coef (all LR pairs)')

# asterisks where n_sig_lr > 0
for i in range(len(beta_all_mat.index)):
    for j in range(len(beta_all_mat.columns)):
        region = beta_all_mat.columns[j]
        ct_pair = beta_all_mat.index[i]
        if region in n_sig_mat.columns:
            n = n_sig_mat.loc[ct_pair, region] if ct_pair in n_sig_mat.index else 0
            if n >= 10:
                marker = '***'
            elif n >= 5:
                marker = '**'
            elif n >= 1:
                marker = '*'
            else:
                marker = ''
            if marker:
                ax.text(j, i, marker, ha='center', va='center',
                        fontsize=5, color='black', fontweight='bold')

ax.set_title(f'Mean age effect per cell type pair x region\n'
             f'All LR pairs shown; * ≥1 sig, ** ≥5 sig, *** ≥10 sig (age_qval<{args.qval_thresh})\n'
             f'Sorted by total significant LR pairs')
plt.tight_layout()
out = os.path.join(PLOT_DIR, 'regression_region_celltype_allbeta_heatmap.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ── ALSO PLOT: n_sig separately ───────────────────────────────────────────────
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

ax.set_title(f'N significant LR pairs per cell type pair x region\n(age_qval<{args.qval_thresh}, min_animals={args.min_animals})')
plt.tight_layout()
out = os.path.join(PLOT_DIR, 'regression_region_celltype_nsig_heatmap.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

print('\nDone.')
