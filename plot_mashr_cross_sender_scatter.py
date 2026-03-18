#!/usr/bin/env python3
"""
plot_mashr_cross_sender_scatter.py
Scatter plot of beta_posterior for the same LR pair across two sender cell types.
Points colored by significance in either/both senders.
Usage:
    python plot_mashr_cross_sender_scatter.py --ct1 Astrocyte --ct2 Glutamatergic
    python plot_mashr_cross_sender_scatter.py --ct1 GABA --ct2 Glutamatergic
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ct1', type=str, required=True)
parser.add_argument('--ct2', type=str, required=True)
parser.add_argument('--lfsr_thresh', type=float, default=0.05)
args = parser.parse_args()

MASHR_DIR = '/scratch/easmit31/cell_cell/results/mashr'
PLOT_DIR  = os.path.join(MASHR_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

def find_results(ct):
    for nan_tag in ['0.55','0.54','0.52','0.50','0.48','0.46']:
        label = f'{ct}_sender_louvain_nanfilt{nan_tag}'
        path  = os.path.join(MASHR_DIR, label, f'mashr_{label}_results.csv')
        if os.path.exists(path):
            return path, label
    return None, None

def load_results(ct):
    for nan_tag in ['0.55', '0.50']:
        label = f'{ct}_sender_louvain_nanfilt{nan_tag}'
        path  = os.path.join(MASHR_DIR, label, f'mashr_{label}_results.csv')
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

res1 = load_results(args.ct1)
res2 = load_results(args.ct2)

if res1 is None or res2 is None:
    print(f'Missing results for {args.ct1} or {args.ct2}')
    exit(1)

# average beta_posterior per lr_pair across all conditions
avg1 = res1.groupby('lr_pair').agg(
    beta1=('beta_posterior','mean'),
    min_lfsr1=('lfsr','min')
).reset_index()
avg2 = res2.groupby('lr_pair').agg(
    beta2=('beta_posterior','mean'),
    min_lfsr2=('lfsr','min')
).reset_index()

merged = avg1.merge(avg2, on='lr_pair')
sig1 = merged['min_lfsr1'] < args.lfsr_thresh
sig2 = merged['min_lfsr2'] < args.lfsr_thresh

colors = np.where(sig1 & sig2, '#d62728',
         np.where(sig1, '#1f77b4',
         np.where(sig2, '#ff7f0e', '#cccccc')))

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(merged['beta1'], merged['beta2'], c=colors, alpha=0.6, s=15)
ax.axhline(0, color='grey', linewidth=0.5)
ax.axvline(0, color='grey', linewidth=0.5)
ax.set_xlabel(f'{args.ct1} sender — mean beta_posterior')
ax.set_ylabel(f'{args.ct2} sender — mean beta_posterior')
ax.set_title(f'Cross-sender beta comparison\n{args.ct1} vs {args.ct2}')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#d62728', markersize=8, label=f'sig in both'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=8, label=f'sig in {args.ct1} only'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=8, label=f'sig in {args.ct2} only'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#cccccc', markersize=8, label='not sig'),
]
ax.legend(handles=legend_elements, fontsize=8)

# label top hits sig in both
both = merged[sig1 & sig2].nlargest(10, 'beta1')
for _, row in both.iterrows():
    ax.annotate(row['lr_pair'], (row['beta1'], row['beta2']),
                fontsize=5, alpha=0.8)

r = merged['beta1'].corr(merged['beta2'])
ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=9)

plt.tight_layout()
out = os.path.join(PLOT_DIR, f'mashr_cross_sender_{args.ct1}_vs_{args.ct2}.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
