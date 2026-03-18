#!/usr/bin/env python3
"""
build_mashr_region_table.py
Builds an intermediate summary table aggregating Louvain-level mashr results
to sender_ct x receiver_ct x region level.
For each combination records:
  - n_sig_louvain: number of significant Louvain combo conditions
  - n_total_louvain: total Louvain combo conditions tested
  - pct_sig: fraction significant
  - mean_beta: mean posterior beta across significant conditions
  - n_sig_lr_pairs: number of unique LR pairs significant
  - dominant_direction: up/down/mixed
Uses both sender and receiver mashr results.
Output: mashr_region_table.csv
"""
import pandas as pd
import numpy as np
import os

MASHR_DIR = '/scratch/easmit31/cell_cell/results/mashr'
ANNOT     = '/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv'

annot = pd.read_csv(ANNOT)[['lr_pair','broad_category']].drop_duplicates('lr_pair')

CTS = ['Astrocyte','GABA','Glutamatergic','Microglia','Oligo','OPC',
       'Vascular','Basket','Cerebellar','Ependymal','Midbrain','MSN']

LFSR_THRESH = 0.05
REGIONS = ['ACC','CN','dlPFC','EC','HIP','IPP','lCb','M1','MB','mdTN','NAc']

def find_results(ct, mode):
    label = ct + '_' + mode + '_louvain_top8perregion'
    path  = os.path.join(MASHR_DIR, label, 'mashr_' + label + '_results.csv')
    if os.path.exists(path):
        return path, label
    for nan_tag in ['0.55','0.54','0.52','0.50','0.48','0.46']:
        label = ct + '_' + mode + '_louvain_nanfilt' + nan_tag
        path  = os.path.join(MASHR_DIR, label, 'mashr_' + label + '_results.csv')
        if os.path.exists(path):
            return path, label
    return None, None

rows = []

for mode in ['sender', 'receiver']:
    for ct in CTS:
        path, label = find_results(ct, mode)
        if path is None:
            print(f'  MISSING {mode}: {ct}')
            continue

        res = pd.read_csv(path)
        # parse condition: source_louvain|target_louvain|region
        parts = res['condition'].str.split('|', expand=True)
        res['source_lou'] = parts[0]
        res['target_lou'] = parts[1]
        res['region']     = parts[2]
        res['source_ct']  = res['source_lou'].str.rsplit('_', n=1).str[0]
        res['target_ct']  = res['target_lou'].str.rsplit('_', n=1).str[0]

        sig = res[res['lfsr'] < LFSR_THRESH]

        for (sender_ct, receiver_ct, region), grp in res.groupby(['source_ct','target_ct','region']):
            sig_grp = grp[grp['lfsr'] < LFSR_THRESH]
            n_sig   = len(sig_grp)
            n_total = len(grp)
            n_sig_lr = sig_grp['lr_pair'].nunique()
            mean_beta = sig_grp['beta_posterior'].mean() if n_sig > 0 else np.nan

            # dominant direction
            if n_sig == 0:
                direction = 'none'
            else:
                n_up   = (sig_grp['beta_posterior'] > 0).sum()
                n_down = (sig_grp['beta_posterior'] < 0).sum()
                if n_up / n_sig > 0.7:   direction = 'up'
                elif n_down / n_sig > 0.7: direction = 'down'
                else:                      direction = 'mixed'

            rows.append({
                'sender_ct':    sender_ct,
                'receiver_ct':  receiver_ct,
                'region':       region,
                'source':       f'{mode}_run',
                'n_sig':        n_sig,
                'n_total':      n_total,
                'pct_sig':      n_sig / n_total if n_total > 0 else 0,
                'mean_beta':    mean_beta,
                'n_sig_lr':     n_sig_lr,
                'direction':    direction,
            })

        print(f'  {ct} {mode}: done')

df = pd.DataFrame(rows)

# keep only rows where both sender and receiver are known cell types
df = df[df['sender_ct'].isin(CTS) & df['receiver_ct'].isin(CTS)]

out = os.path.join(MASHR_DIR, 'mashr_region_table_top8.csv')
df.to_csv(out, index=False)
print(f'\nSaved: {out}')
print(f'Shape: {df.shape}')
print(f'\nRegions covered: {sorted(df["region"].unique())}')
print(f'Sender CTs: {sorted(df["sender_ct"].unique())}')
