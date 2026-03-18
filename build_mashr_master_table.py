#!/usr/bin/env python3
"""
build_mashr_master_table.py
Builds a master table of LR pair x sender->receiver cell type pairs.
For each combination, records:
  - n_conditions_sig: number of Louvain combo x region conditions significant
  - mean_beta_posterior: mean posterior effect size across significant conditions
  - n_conditions_total: total conditions tested
  - pct_sig: fraction of conditions significant
  - sharing_tier: global/broad/intermediate/specific_few/specific_one/not_sig
Uses both sender and receiver mashr results.
Output: mashr_master_table.csv
Usage:
    python build_mashr_master_table.py
"""
import pandas as pd
import numpy as np
import os

MASHR_DIR = '/scratch/easmit31/cell_cell/results/mashr'
ANNOT     = '/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv'
PLOT_DIR  = os.path.join(MASHR_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

annot = pd.read_csv(ANNOT)[['lr_pair','broad_category','classification']].drop_duplicates('lr_pair')

CTS = ['Astrocyte','GABA','Glutamatergic','Microglia','Oligo','OPC',
       'Vascular','Basket','Cerebellar','Ependymal','Midbrain','MSN']

LFSR_THRESH = 0.05

def find_results(ct, mode):
    for nan_tag in ['0.55','0.54','0.52','0.50','0.48','0.46']:
        label = f'{ct}_{mode}_louvain_nanfilt{nan_tag}'
        path  = os.path.join(MASHR_DIR, label, f'mashr_{label}_results.csv')
        if os.path.exists(path):
            return path, label
    return None, None

def get_sharing_tier(n, n_conds):
    if n == 0:           return 'not_sig'
    if n == n_conds:     return 'global'
    if n > n_conds // 2: return 'broad'
    if n >= 6:           return 'intermediate'
    if n >= 2:           return 'specific_few'
    return 'specific_one'

rows = []

# ── SENDER RESULTS ────────────────────────────────────────────────────────────
print("Processing sender results...")
for sender_ct in CTS:
    path, label = find_results(sender_ct, 'sender')
    if path is None:
        print(f'  MISSING sender: {sender_ct}')
        continue

    res = pd.read_csv(path)
    n_conds = res['condition'].nunique()

    # parse receiver cell type from condition: source_lou|target_lou|region
    # condition format for sender mode: source_louvain|target_louvain|region
    res['receiver_ct'] = res['condition'].str.split('|').str[1].str.rsplit('_', n=1).str[0]

    sig = res[res['lfsr'] < LFSR_THRESH]

    # group by lr_pair x receiver_ct
    for (lr_pair, receiver_ct), grp in res.groupby(['lr_pair','receiver_ct']):
        sig_grp = grp[grp['lfsr'] < LFSR_THRESH]
        n_sig = len(sig_grp)
        n_total = len(grp)
        mean_beta = sig_grp['beta_posterior'].mean() if n_sig > 0 else np.nan
        mean_beta_all = grp['beta_posterior'].mean()

        rows.append({
            'lr_pair':        lr_pair,
            'sender_ct':      sender_ct,
            'receiver_ct':    receiver_ct,
            'source':         'sender_run',
            'n_sig':          n_sig,
            'n_total':        n_total,
            'pct_sig':        n_sig / n_total if n_total > 0 else 0,
            'mean_beta_sig':  mean_beta,
            'mean_beta_all':  mean_beta_all,
            'tier':           get_sharing_tier(n_sig, n_total),
        })

    print(f'  {sender_ct} sender: done')

# ── RECEIVER RESULTS ──────────────────────────────────────────────────────────
print("Processing receiver results...")
for receiver_ct in CTS:
    path, label = find_results(receiver_ct, 'receiver')
    if path is None:
        print(f'  MISSING receiver: {receiver_ct}')
        continue

    res = pd.read_csv(path)
    n_conds = res['condition'].nunique()

    # condition format for receiver mode: source_louvain|target_louvain|region
    res['sender_ct'] = res['condition'].str.split('|').str[0].str.rsplit('_', n=1).str[0]

    for (lr_pair, sender_ct), grp in res.groupby(['lr_pair','sender_ct']):
        sig_grp = grp[grp['lfsr'] < LFSR_THRESH]
        n_sig = len(sig_grp)
        n_total = len(grp)
        mean_beta = sig_grp['beta_posterior'].mean() if n_sig > 0 else np.nan
        mean_beta_all = grp['beta_posterior'].mean()

        rows.append({
            'lr_pair':        lr_pair,
            'sender_ct':      sender_ct,
            'receiver_ct':    receiver_ct,
            'source':         'receiver_run',
            'n_sig':          n_sig,
            'n_total':        n_total,
            'pct_sig':        n_sig / n_total if n_total > 0 else 0,
            'mean_beta_sig':  mean_beta,
            'mean_beta_all':  mean_beta_all,
            'tier':           get_sharing_tier(n_sig, n_total),
        })

    print(f'  {receiver_ct} receiver: done')

df = pd.DataFrame(rows)

# merge annotations
df = df.merge(annot, on='lr_pair', how='left')
df['broad_category'] = df['broad_category'].fillna('Unknown')

# save
out = os.path.join(MASHR_DIR, 'mashr_master_table.csv')
df.to_csv(out, index=False)
print(f'\nSaved: {out}')
print(f'Shape: {df.shape}')
print(f'\nRows by source:')
print(df['source'].value_counts())
print(f'\nRows by tier:')
print(df['tier'].value_counts())
