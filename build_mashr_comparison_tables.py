#!/usr/bin/env python3
"""
build_mashr_comparison_tables.py
Builds nicely formatted comparison tables for all 4 mashr run types:
  - sender nanfilt
  - sender top8perregion
  - receiver nanfilt
  - receiver top8perregion
Saves both CSV and PNG table images.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

MASHR_DIR = '/scratch/easmit31/cell_cell/results/mashr'
PLOT_DIR  = os.path.join(MASHR_DIR, 'plots_comparison')
os.makedirs(PLOT_DIR, exist_ok=True)

CTS = ['Astrocyte','GABA','Glutamatergic','Microglia','Oligo','OPC',
       'Vascular','Basket','Cerebellar','Ependymal','Midbrain','MSN']
REGIONS = ['ACC','CN','dlPFC','EC','HIP','IPP','lCb','M1','MB','mdTN','NAc']
LFSR = 0.05

def find_nanfilt(ct, mode):
    for nan_tag in ['0.55','0.54','0.52','0.50','0.48','0.46']:
        label = ct + '_' + mode + '_louvain_nanfilt' + nan_tag
        path  = os.path.join(MASHR_DIR, label, 'mashr_' + label + '_results.csv')
        if os.path.exists(path): return path, label
    return None, None

def find_top8(ct, mode):
    label = ct + '_' + mode + '_louvain_top8perregion'
    path  = os.path.join(MASHR_DIR, label, 'mashr_' + label + '_results.csv')
    return (path, label) if os.path.exists(path) else (None, None)

def get_sharing_stats(path):
    if path is None: return None
    res = pd.read_csv(path)
    sig = res[res['lfsr'] < LFSR]
    lr_counts = sig['lr_pair'].value_counts()
    n_conds = res['condition'].nunique()
    parts = res['condition'].str.split('|', expand=True)
    res['region'] = parts[2]
    region_counts = res.groupby('region')['condition'].nunique()
    return {
        'n_conditions': n_conds,
        'n_lr_pairs':   res['lr_pair'].nunique(),
        'sig_all':      (lr_counts == n_conds).sum(),
        'sig_gt50pct':  (lr_counts > n_conds//2).sum(),
        'sig_6_20':     ((lr_counts >= 6) & (lr_counts <= 20)).sum(),
        'sig_2_5':      ((lr_counts >= 2) & (lr_counts <= 5)).sum(),
        'sig_1':        (lr_counts == 1).sum(),
        'total_sig':    len(lr_counts),
        'regions':      {r: int(region_counts.get(r, 0)) for r in REGIONS},
    }

def save_table_png(df, title, out_path, col_width=1.2):
    fig, ax = plt.subplots(figsize=(max(12, len(df.columns)*col_width + 2),
                                    max(4, len(df)*0.4 + 1.5)))
    ax.axis('off')
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc='center',
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))

    # color header
    for j in range(len(df.columns)):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # color row labels
    for i in range(len(df)):
        tbl[i+1, -1].set_facecolor('#ecf0f1')

    # alternating row colors
    for i in range(len(df)):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                tbl[i+1, j].set_facecolor('#f8f9fa')

    plt.title(title, fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

# ── TABLE 1: Sharing distribution per run type ────────────────────────────────
for mode in ['sender', 'receiver']:
    for run_type, find_fn in [('nanfilt', find_nanfilt), ('top8perregion', find_top8)]:
        rows = []
        for ct in CTS:
            path, label = find_fn(ct, mode)
            stats = get_sharing_stats(path)
            if stats is None:
                rows.append({'sender': ct, 'n_conditions': 'MISSING',
                             'n_lr_pairs': '', 'sig_all': '', 'sig_gt50pct': '',
                             'sig_6_20': '', 'sig_2_5': '', 'sig_1': '', 'total_sig': ''})
                continue
            rows.append({
                'cell_type':    ct,
                'n_conditions': stats['n_conditions'],
                'n_lr_pairs':   stats['n_lr_pairs'],
                'sig_all':      stats['sig_all'],
                'sig_>50%':     stats['sig_gt50pct'],
                'sig_6-20':     stats['sig_6_20'],
                'sig_2-5':      stats['sig_2_5'],
                'sig_1':        stats['sig_1'],
                'total_sig':    stats['total_sig'],
            })

        df = pd.DataFrame(rows).set_index('cell_type')
        tag = f'{mode}_{run_type}'

        # save CSV
        csv_out = os.path.join(PLOT_DIR, f'sharing_distribution_{tag}.csv')
        df.to_csv(csv_out)
        print(f'Saved: {csv_out}')

        # save PNG
        save_table_png(df, f'mashr sharing distribution — {mode} {run_type}',
                      os.path.join(PLOT_DIR, f'sharing_distribution_{tag}.png'))

# ── TABLE 2: Regional coverage per run type ───────────────────────────────────
for mode in ['sender', 'receiver']:
    for run_type, find_fn in [('nanfilt', find_nanfilt), ('top8perregion', find_top8)]:
        rows = []
        for ct in CTS:
            path, label = find_fn(ct, mode)
            stats = get_sharing_stats(path)
            if stats is None:
                row = {'cell_type': ct, 'n_conditions': 'MISSING'}
                row.update({r: '' for r in REGIONS})
                rows.append(row)
                continue
            row = {'cell_type': ct, 'n_conditions': stats['n_conditions']}
            row.update({r: stats['regions'][r] if stats['regions'][r] > 0 else '' for r in REGIONS})
            rows.append(row)

        df = pd.DataFrame(rows).set_index('cell_type')
        tag = f'{mode}_{run_type}'

        csv_out = os.path.join(PLOT_DIR, f'regional_coverage_{tag}.csv')
        df.to_csv(csv_out)
        print(f'Saved: {csv_out}')

        save_table_png(df, f'mashr regional coverage — {mode} {run_type}',
                      os.path.join(PLOT_DIR, f'regional_coverage_{tag}.png'),
                      col_width=0.9)

# ── TABLE 3: Overlap between nanfilt and top8 ─────────────────────────────────
for mode in ['sender', 'receiver']:
    rows = []
    for ct in CTS:
        p1, _ = find_nanfilt(ct, mode)
        p2, _ = find_top8(ct, mode)
        if p1 is None or p2 is None:
            rows.append({'cell_type': ct, 'n_sig_nanfilt': 'MISSING',
                         'n_sig_top8': '', 'overlap': '', 'pct_overlap': '',
                         'direction_agree': ''})
            continue

        r1 = pd.read_csv(p1)
        r2 = pd.read_csv(p2)

        sig1 = set(r1[r1['lfsr'] < LFSR]['lr_pair'].unique())
        sig2 = set(r2[r2['lfsr'] < LFSR]['lr_pair'].unique())
        overlap = sig1 & sig2

        beta1 = r1[r1['lr_pair'].isin(overlap)].groupby('lr_pair')['beta_posterior'].mean()
        beta2 = r2[r2['lr_pair'].isin(overlap)].groupby('lr_pair')['beta_posterior'].mean()
        common = beta1.index.intersection(beta2.index)
        agree = (beta1[common] * beta2[common] > 0).mean() if len(common) > 0 else np.nan
        pct = len(overlap) / len(sig1 | sig2) if len(sig1 | sig2) > 0 else 0

        rows.append({
            'cell_type':       ct,
            'n_sig_nanfilt':   len(sig1),
            'n_sig_top8':      len(sig2),
            'overlap':         len(overlap),
            'pct_overlap':     f'{pct:.1%}',
            'direction_agree': f'{agree:.1%}' if not np.isnan(agree) else 'N/A',
        })

    df = pd.DataFrame(rows).set_index('cell_type')

    csv_out = os.path.join(PLOT_DIR, f'overlap_nanfilt_vs_top8_{mode}.csv')
    df.to_csv(csv_out)
    print(f'Saved: {csv_out}')

    save_table_png(df, f'nanfilt vs top8perregion overlap — {mode}',
                  os.path.join(PLOT_DIR, f'overlap_nanfilt_vs_top8_{mode}.png'))

print('\nAll tables saved.')
