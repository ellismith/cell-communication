#!/usr/bin/env python3
"""
plot_mashr_scatter_sanity.py
Scatter plots of lr_means vs age for specific LR pair x condition combinations.
Shows raw data points colored by sex, OLS regression line, and mashr-shrunk beta.
Used for sanity checking mashr results against raw per-animal data.
Usage:
    python plot_mashr_scatter_sanity.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
from scipy import stats

MASHR_DIR = '/scratch/easmit31/cell_cell/results/mashr'
MATRIX_DIR = '/scratch/easmit31/cell_cell/results/lr_matrices'
PLOT_DIR  = os.path.join(MASHR_DIR, 'plots', 'scatter_sanity')
os.makedirs(PLOT_DIR, exist_ok=True)

# LR pairs and conditions to plot
# format: (lr_pair, sender_ct, [(condition, label), ...])
TO_PLOT = {
    'SLC1A1|GRM3': {
        'sender_ct': 'GABA',
        'sig': [
            'GABA_6|MSN_9|CN',
            'GABA_6|Ependymal_7|CN',
            'GABA_5|Ependymal_7|CN',
            'GABA_10|Glutamatergic_1|dlPFC',
            'GABA_3|Glutamatergic_2|EC',
        ],
        'nonsig': [
            'GABA_8|Glutamatergic_11|HIP',
            'GABA_8|GABA_8|HIP',
            'GABA_8|Microglia_9|EC',
        ]
    },
    'NTN1|UNC5C': {
        'sender_ct': 'GABA',
        'sig': [
            'GABA_8|Astrocyte_7|EC',
            'GABA_5|MSN_9|CN',
            'GABA_6|Vascular_8|CN',
            'GABA_10|GABA_3|EC',
            'GABA_8|GABA_3|EC',
        ],
        'nonsig': [
            'GABA_0|GABA_8|EC',
            'GABA_8|GABA_9|HIP',
            'GABA_8|GABA_4|EC',
        ]
    }
}

# load mashr results for GABA sender
mashr = pd.read_csv(f'{MASHR_DIR}/GABA_sender_louvain_nanfilt0.55/mashr_GABA_sender_louvain_nanfilt0.55_results.csv')

# load per-animal data from lr matrices for each region
def get_region_matrix(region):
    path = f'{MATRIX_DIR}/{region}_nothresh_minage1p0_matrix.csv'
    if not os.path.exists(path):
        return None
    mat = pd.read_csv(path, index_col=0, low_memory=False)
    age_row = mat.loc['age'].astype(float)
    sex_row = mat.loc['sex']
    mat = mat.drop(index=['age','sex'])
    return mat, age_row, sex_row

# cache matrices
region_cache = {}
for region in ['ACC','CN','dlPFC','EC','HIP','IPP','lCb','M1','MB','mdTN','NAc']:
    result = get_region_matrix(region)
    if result is not None:
        region_cache[region] = result

for lr_pair, info in TO_PLOT.items():
    all_conditions = [('sig', c) for c in info['sig']] + [('nonsig', c) for c in info['nonsig']]
    n_plots = len(all_conditions)
    ncols = 4
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3.5))
    axes = axes.flatten()

    for idx, (status, condition) in enumerate(all_conditions):
        ax = axes[idx]
        parts = condition.split('|')
        source_lou = parts[0]
        target_lou = parts[1]
        region     = parts[2]

        # get mashr results for this condition
        m = mashr[(mashr['lr_pair']==lr_pair) & (mashr['condition']==condition)]
        if len(m) == 0:
            ax.set_title(f'{condition}\n(no mashr result)', fontsize=7)
            ax.axis('off')
            continue

        beta_post = m['beta_posterior'].values[0]
        beta_orig = m['beta_original'].values[0]
        lfsr_val  = m['lfsr'].values[0]

        # get raw per-animal lr_means from matrix
        if region not in region_cache:
            ax.set_title(f'{condition}\n(no matrix)', fontsize=7)
            ax.axis('off')
            continue

        mat, age_row, sex_row = region_cache[region]

        # interaction key
        ligand, receptor = lr_pair.split('|')
        interaction = f'{source_lou}|{target_lou}|{ligand}|{receptor}'

        if interaction not in mat.index:
            ax.set_title(f'{condition}\n(interaction not in matrix)', fontsize=7)
            ax.axis('off')
            continue

        row = mat.loc[interaction].dropna()
        ages = age_row[row.index].astype(float)
        sexes = sex_row[row.index]
        lr_vals = row.astype(float)

        if len(lr_vals) < 3:
            ax.set_title(f'{condition}\nn={len(lr_vals)} (too few)', fontsize=7)
            ax.axis('off')
            continue

        # scatter colored by sex
        colors = ['#e41a1c' if s=='F' else '#377eb8' for s in sexes]
        ax.scatter(ages, lr_vals, c=colors, s=30, alpha=0.8, zorder=3)

        # OLS regression line
        slope, intercept, r, p, _ = stats.linregress(ages, lr_vals)
        x_line = np.linspace(ages.min(), ages.max(), 100)
        ax.plot(x_line, intercept + slope*x_line, color='black',
                linewidth=1.5, label=f'OLS β={slope:.4f}')

        # mashr posterior line (same intercept, different slope)
        ax.plot(x_line, intercept + beta_post*x_line, color='purple',
                linewidth=1.5, linestyle='--', label=f'mashr β={beta_post:.4f}')

        # formatting
        border_color = '#2ca02c' if status == 'sig' else '#999999'
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)

        ax.set_xlabel('Age', fontsize=7)
        ax.set_ylabel('lr_means', fontsize=7)
        ax.tick_params(labelsize=6)
        title = f'{condition}\nlfsr={lfsr_val:.3f} n={len(lr_vals)}'
        ax.set_title(title, fontsize=6.5,
                     color='#2ca02c' if status=='sig' else '#999999')
        ax.legend(fontsize=5.5, loc='best')

    # hide unused axes
    for idx in range(len(all_conditions), len(axes)):
        axes[idx].axis('off')

    # legend for sex
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#e41a1c', markersize=8, label='Female'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#377eb8', markersize=8, label='Male'),
        Line2D([0],[0], color='black', linewidth=1.5, label='OLS beta'),
        Line2D([0],[0], color='purple', linewidth=1.5, linestyle='--', label='mashr beta'),
    ], loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'{lr_pair} — GABA sender\nGreen border=significant (lfsr<0.05), Grey=non-significant',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    safe_lr = lr_pair.replace('|','_')
    out = os.path.join(PLOT_DIR, f'scatter_sanity_{safe_lr}_GABA_sender.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')
