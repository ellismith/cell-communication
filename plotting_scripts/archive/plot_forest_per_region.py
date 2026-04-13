#!/usr/bin/env python3
"""
plot_forest_per_region.py
Forest plots per region showing age effects on LR pairs.
One plot per region, rows = LR pairs grouped by sender cell type.
Points = mean age_coef across Louvain combos, error bars = stderr.
Color = sender cell type.
Built from whole-region regression outputs.
Usage:
    python plot_forest_per_region.py
    python plot_forest_per_region.py --top_n 20 --qval_thresh 0.05
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--top_n', type=int, default=15,
                    help='Top N LR pairs per sender cell type to show')
parser.add_argument('--qval_thresh', type=float, default=0.05)
parser.add_argument('--min_animals', type=int, default=10)
args = parser.parse_args()

BASE     = '/scratch/easmit31/cell_cell/results/within_region_analysis'
ANNOT    = '/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv'
PLOT_DIR = '/scratch/easmit31/cell_cell/results/regression_plots/forest_plots'
os.makedirs(PLOT_DIR, exist_ok=True)

REGIONS = ['ACC','CN','dlPFC','EC','HIP','IPP','lCb','M1','MB','mdTN','NAc']
CTS = ['Astrocyte','GABA','Glutamatergic','Microglia','Oligo','OPC',
       'Vascular','Basket','Cerebellar','Ependymal','Midbrain','MSN']

CT_COLORS = {
    'Astrocyte':     '#e6194b',
    'GABA':          '#3cb44b',
    'Glutamatergic': '#4363d8',
    'Microglia':     '#f58231',
    'Oligo':         '#911eb4',
    'OPC':           '#42d4f4',
    'Vascular':      '#f032e6',
    'Basket':        '#bfef45',
    'Cerebellar':    '#fabed4',
    'Ependymal':     '#469990',
    'Midbrain':      '#dcbeff',
    'MSN':           '#9a6324',
}

annot = pd.read_csv(ANNOT)[['lr_pair','broad_category']].drop_duplicates('lr_pair')

for region in REGIONS:
    rlow = region.lower()
    path = f'{BASE}/regression_{region}/whole_{rlow}_age_sex_regression.csv'
    if not os.path.exists(path):
        print(f'  MISSING: {region}')
        continue

    print(f'  Processing {region}...')
    df = pd.read_csv(path, engine='c')
    df = df[df['n_animals'] >= args.min_animals]

    parts = df['interaction'].str.split('|', expand=True)
    df['source_ct'] = parts[0].str.rsplit('_', n=1).str[0]
    df['target_ct'] = parts[1].str.rsplit('_', n=1).str[0]
    df['lr_pair']   = parts[2] + '|' + parts[3]
    df = df[df['source_ct'].isin(CTS) & df['target_ct'].isin(CTS)]
    df = df.merge(annot, on='lr_pair', how='left')
    df['broad_category'] = df['broad_category'].fillna('Unknown')

    # aggregate to lr_pair x source_ct level
    # mean age_coef, pooled stderr, min qval across Louvain combos
    agg = df.groupby(['lr_pair','source_ct','broad_category']).agg(
        mean_beta=('age_coef','mean'),
        se=('age_stderr', lambda x: np.sqrt(np.mean(x**2))),  # RMS SE
        min_qval=('age_qval','min'),
        n_louvain=('interaction','count')
    ).reset_index()

    # keep only significant
    sig = agg[agg['min_qval'] < args.qval_thresh]
    if len(sig) == 0:
        print(f'  {region}: no significant interactions')
        continue

    # for each sender CT, take top N by |mean_beta|
    rows = []
    ct_order = []
    for ct in CTS:
        ct_sig = sig[sig['source_ct']==ct].copy()
        if len(ct_sig) == 0:
            continue
        ct_top = ct_sig.nlargest(args.top_n, 'mean_beta').append(
                 ct_sig.nsmallest(args.top_n, 'mean_beta')).drop_duplicates()
        ct_top = ct_top.sort_values('mean_beta', ascending=True)
        rows.append(ct_top)
        ct_order.append(ct)

    if not rows:
        continue

    plot_df = pd.concat(rows, ignore_index=True)

    # ── PLOT ──────────────────────────────────────────────────────────────────
    n_rows = len(plot_df)
    fig, ax = plt.subplots(figsize=(10, max(6, n_rows * 0.25 + 2)))

    # plot each row
    y_pos = 0
    y_ticks = []
    y_labels = []
    separator_positions = []

    prev_ct = None
    for _, row in plot_df.iterrows():
        ct = row['source_ct']
        if ct != prev_ct and prev_ct is not None:
            separator_positions.append(y_pos - 0.5)
            y_pos += 0.5  # gap between cell types
        prev_ct = ct

        color = CT_COLORS.get(ct, '#888888')
        beta  = row['mean_beta']
        se    = row['se']

        # point and error bar
        ax.errorbar(beta, y_pos, xerr=se*1.96,
                   fmt='o', color=color, markersize=4,
                   capsize=2, capthick=1, linewidth=1,
                   alpha=0.85, zorder=3)

        # significance marker
        if row['min_qval'] < 0.001:   marker = '***'
        elif row['min_qval'] < 0.01:  marker = '**'
        else:                          marker = '*'

        ax.text(beta + se*1.96 + 0.0001, y_pos, marker,
                va='center', ha='left', fontsize=5,
                color=color)

        # label: lr_pair [category]
        label = f"{row['lr_pair']} [{row['broad_category'][:12]}]"
        y_ticks.append(y_pos)
        y_labels.append(label)
        y_pos += 1

    # formatting
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel('Mean age coefficient (age_coef)', fontsize=9)

    # separator lines between cell types
    for sep in separator_positions:
        ax.axhline(sep, color='grey', linewidth=0.5, alpha=0.4, linestyle='-')

    # cell type labels on right
    prev_ct = None
    ct_start = 0
    ct_positions = {}
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ct = row['source_ct']
        if ct != prev_ct:
            if prev_ct is not None:
                ct_positions[prev_ct] = (ct_start, i-1)
            ct_start = i
            prev_ct = ct
    if prev_ct:
        ct_positions[prev_ct] = (ct_start, len(plot_df)-1)

    for ct, (start, end) in ct_positions.items():
        mid_y = (y_ticks[start] + y_ticks[end]) / 2
        ax.text(ax.get_xlim()[1] * 1.02, mid_y, ct,
                va='center', ha='left', fontsize=7,
                color=CT_COLORS.get(ct,'#888'),
                fontweight='bold')
        # colored bar on right
        ax.barh(range(start, end+1),
                [ax.get_xlim()[1] * 0.005] * (end-start+1),
                left=ax.get_xlim()[1] * 1.08,
                height=0.8,
                color=CT_COLORS.get(ct,'#888'),
                alpha=0.8)

    ax.set_title(f'{region} — Top LR pairs by age effect\n'
                 f'Grouped by sender cell type, sorted by effect size\n'
                 f'Error bars = 95% CI, * q<0.05, ** q<0.01, *** q<0.001',
                 fontsize=9)

    # legend
    legend_elements = [mpatches.Patch(facecolor=CT_COLORS[ct], label=ct)
                       for ct in ct_order]
    ax.legend(handles=legend_elements, loc='lower right',
              fontsize=6, ncol=2)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'forest_{region}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')

print('\nDone.')
