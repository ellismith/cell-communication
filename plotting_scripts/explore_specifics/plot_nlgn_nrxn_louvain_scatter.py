#!/usr/bin/env python3
"""
Scatter plots of lr_means vs age for NLGN1|NRXN1, Astrocyte->GABA in EC.
One subplot per Louvain combo. Sample 10 significant + 10 non-significant.
Significant combos marked with * in title.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from matplotlib.patches import Patch

MATRIX_FILE = Path("/scratch/easmit31/cell_cell/results/lr_matrices_corrected/EC_nothresh_minage1p0_matrix.csv")
REG_FILE    = Path("/scratch/easmit31/cell_cell/results/within_region_analysis_corrected/regression_EC/whole_ec_age_sex_regression.csv")
OUT         = Path("/scratch/easmit31/cell_cell/results/within_region_analysis_corrected/regression_EC/nlgn_nrxn_louvain_scatters")
OUT.mkdir(parents=True, exist_ok=True)

LR_PAIR  = "NLGN1|NRXN1"
Q_THRESH = 0.05

# load regression in chunks
print("Loading regression...")
chunks = []
for chunk in pd.read_csv(REG_FILE, chunksize=500000):
    parts = chunk['interaction'].str.split('|', expand=True)
    chunk['sender']         = parts[0]
    chunk['receiver']       = parts[1]
    chunk['lr_pair']        = parts[2] + '|' + parts[3]
    chunk['sender_broad']   = chunk['sender'].str.replace(r'_\d+$','',regex=True)
    chunk['receiver_broad'] = chunk['receiver'].str.replace(r'_\d+$','',regex=True)
    sub = chunk[(chunk['sender_broad']=='Astrocyte') &
                (chunk['receiver_broad']=='GABA') &
                (chunk['lr_pair']==LR_PAIR)]
    if len(sub):
        chunks.append(sub)

reg = pd.concat(chunks)
print(f"Total Louvain combos: {len(reg)}, Significant: {(reg['age_qval']<Q_THRESH).sum()}")

sig     = reg[reg['age_qval'] < Q_THRESH]
not_sig = reg[reg['age_qval'] >= Q_THRESH]

np.random.seed(42)
sampled_sig     = sig.sample(min(10, len(sig)))
sampled_not_sig = not_sig.sample(min(10, len(not_sig)))
sampled = pd.concat([sampled_sig, sampled_not_sig])
sampled['significant'] = sampled['age_qval'] < Q_THRESH

print(f"Plotting {len(sampled)} combos ({len(sampled_sig)} sig, {len(sampled_not_sig)} not sig)")

# load matrix
print("Loading matrix...")
mat = pd.read_csv(MATRIX_FILE, index_col=0, low_memory=False)
age_row = mat.loc['age'].astype(float)
sex_row = mat.loc['sex']
mat     = mat.drop(index=['age','sex'])

n = len(sampled)
ncols = 5
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
axes = axes.flatten()

for ax, (_, row) in zip(axes, sampled.iterrows()):
    sender, receiver = row['sender'], row['receiver']
    key = f"{sender}|{receiver}|NLGN1|NRXN1"

    if key not in mat.index:
        ax.set_visible(False)
        continue

    vals = mat.loc[key]
    df_plot = pd.DataFrame({
        'age':     age_row,
        'lr_means': vals.astype(float),
        'sex':     sex_row
    }).dropna()

    colors = df_plot['sex'].map({'M':'#1f77b4','F':'#e91e63'})
    ax.scatter(df_plot['age'], df_plot['lr_means'], c=colors, alpha=0.6, s=25)

    slope, intercept, r, p, _ = stats.linregress(df_plot['age'], df_plot['lr_means'])
    x = np.linspace(df_plot['age'].min(), df_plot['age'].max(), 100)
    ax.plot(x, slope*x + intercept, 'k-', lw=1.5)

    sig_marker = '★' if row['significant'] else ''
    ax.set_title(f"{sig_marker}{sender}→{receiver}\nr={r:.2f} q={row['age_qval']:.2e}",
                 fontsize=8, color='red' if row['significant'] else 'black')
    ax.set_xlabel("Age", fontsize=7)
    ax.set_ylabel("lr_means", fontsize=7)
    ax.tick_params(labelsize=7)

for ax in axes[n:]:
    ax.set_visible(False)

fig.suptitle(f"EC — NLGN1|NRXN1 — Astrocyte→GABA\n10 significant (★) + 10 non-significant Louvain combos",
             fontsize=12, fontweight='bold')
fig.legend(handles=[Patch(color='#e91e63', label='F'), Patch(color='#1f77b4', label='M')],
           loc='lower right', fontsize=9)
plt.tight_layout()

fname = OUT / "EC_NLGN1_NRXN1_Astrocyte_GABA_sampled_louvains.png"
fig.savefig(fname, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {fname}")
