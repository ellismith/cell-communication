#!/usr/bin/env python3
"""
plot_mashr_sharing_table.py
Summary table of mashr sharing distribution per sender cell type.
Shows: sig in all conditions, >50%, 6-20, 2-5, exactly 1, total sig.
Usage:
    python plot_mashr_sharing_table.py
    python plot_mashr_sharing_table.py --sender_ct Astrocyte
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
args = parser.parse_args()

MASHR_DIR = '/scratch/easmit31/cell_cell/results/mashr'
PLOT_DIR  = os.path.join(MASHR_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

CTS = ['Astrocyte','GABA','Glutamatergic','Microglia','Oligo','OPC',
       'Vascular','Basket','Cerebellar','Ependymal','Midbrain','MSN']
if args.sender_ct:
    CTS = [args.sender_ct]

def find_results(ct):
    for nan_tag in ['0.55','0.54','0.52','0.50','0.48','0.46']:
        label = f'{ct}_sender_louvain_nanfilt{nan_tag}'
        path  = os.path.join(MASHR_DIR, label, f'mashr_{label}_results.csv')
        if os.path.exists(path):
            return path, label
    return None, None

rows = []
for ct in CTS:
    path, label = find_results(ct)
    if path is None:
        print(f'  MISSING: {ct}')
        continue

    res = pd.read_csv(path)
    sig = res[res['lfsr'] < args.lfsr_thresh]
    n_conds = res['condition'].nunique()
    lr_counts = sig['lr_pair'].value_counts()

    rows.append({
        'sender':       ct,
        'n_conditions': n_conds,
        'n_lr_pairs':   res['lr_pair'].nunique(),
        'sig_all':      (lr_counts == n_conds).sum(),
        'sig_gt50pct':  (lr_counts > n_conds // 2).sum(),
        'sig_6_20':     ((lr_counts >= 6) & (lr_counts <= 20)).sum(),
        'sig_2_5':      ((lr_counts >= 2) & (lr_counts <= 5)).sum(),
        'sig_1':        (lr_counts == 1).sum(),
        'total_sig':    len(lr_counts),
    })
    print(f'  {ct}: done ({label})')

df = pd.DataFrame(rows)
print(df.to_string(index=False))

fig, ax = plt.subplots(figsize=(14, max(3, len(df) * 0.5 + 1)))
ax.axis('off')
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.auto_set_column_width(col=list(range(len(df.columns))))
plt.title(f'mashr sharing distribution (lfsr<{args.lfsr_thresh})', fontsize=12, pad=20)
plt.tight_layout()
out = os.path.join(PLOT_DIR, 'mashr_sharing_table.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')

csv_out = os.path.join(PLOT_DIR, 'mashr_sharing_table.csv')
df.to_csv(csv_out, index=False)
print(f'Saved: {csv_out}')
