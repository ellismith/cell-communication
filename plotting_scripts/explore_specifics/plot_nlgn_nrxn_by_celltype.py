#!/usr/bin/env python3
"""
Scatter plots of lr_means vs age for NLGN1|NRXN1, NLGN1|NRXN2, NLGN1|NRXN3
Split by broad cell type sender->receiver combo.
Only significant LR pairs (age_qval < 0.05 in regression output).
Each point = one animal x one Louvain combo.
Louvain pairs listed in subtitle.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from matplotlib.patches import Patch
import glob
import re

MATRIX_DIR = Path("/Users/elliotsmith/Desktop/LR_matrices")
REG_DIR    = Path("/Users/elliotsmith/Desktop/regression_results")
OUT        = Path("/Users/elliotsmith/Desktop/nlgn_nrxn_scatters_by_celltype")
OUT.mkdir(parents=True, exist_ok=True)

LR_PAIRS = ["NLGN1|NRXN1", "NLGN1|NRXN2", "NLGN1|NRXN3"]
REGIONS  = ["ACC", "CN", "dlPFC", "EC", "HIP", "IPP", "lCb", "M1", "MB", "mdTN", "NAc"]
REGION_LOWER = {
    "ACC": "acc", "CN": "cn", "dlPFC": "dlpfc", "EC": "ec", "HIP": "hip",
    "IPP": "ipp", "lCb": "lcb", "M1": "m1", "MB": "mb", "mdTN": "mdtn", "NAc": "nac",
}

Q_THRESH = 0.05

BROAD_ORDER = ["Glutamatergic", "GABA", "Astrocyte", "Oligo", "OPC", "Microglia", "Vascular", "Basket", "Cerebellar", "MSN", "Midbrain", "Ependymal", "Other"]

def broad_class(s):
    for b in BROAD_ORDER:
        if s.startswith(b):
            return b
    return "Other"

for region in REGIONS:
    fpath = MATRIX_DIR / f"{region}_nothresh_minage1p0_matrix.csv"
    region_lower = REGION_LOWER.get(region, region.lower())
    reg_files = list((REG_DIR / f"regression_{region_lower}").glob(f"whole_{region_lower}_age_sex_regression*.csv"))

    if not fpath.exists():
        print(f"[skip] {region} -- matrix not found")
        continue
    if not reg_files:
        print(f"[skip] {region} -- regression not found")
        continue

    mat = pd.read_csv(fpath, index_col=0, low_memory=False)
    age_row = mat.loc['age'].astype(float)
    sex_row = mat.loc['sex']
    mat = mat.drop(index=['age', 'sex'])

    reg = pd.read_csv(reg_files[0])
    split = reg['interaction'].str.split('|', expand=True)
    reg['sender']   = split[0]
    reg['receiver'] = split[1]
    reg['lr_pair']  = split[2] + '|' + split[3]
    reg['sender_broad']   = reg['sender'].str.replace(r'_\d+$', '', regex=True)
    reg['receiver_broad'] = reg['receiver'].str.replace(r'_\d+$', '', regex=True)

    # filter to NLGN/NRXN sig pairs
    reg_nlgn = reg[reg['lr_pair'].isin(LR_PAIRS) & (reg['age_qval'] < Q_THRESH)]
    if reg_nlgn.empty:
        print(f"[skip] {region} -- no significant NLGN/NRXN pairs")
        continue

    # get unique broad combos
    broad_combos = reg_nlgn.groupby(['sender_broad', 'receiver_broad'])
    print(f"\n{region}: {len(broad_combos)} broad combos, {len(reg_nlgn)} sig interactions")

    for (sb, rb), group in broad_combos:
        combo_label = f"{sb}→{rb}"
        sig_interactions = set(group['sender'] + '|' + group['receiver'] + '|' + group['lr_pair'])
        sig_lr_pairs = group['lr_pair'].unique()
        louvain_pairs = sorted(set(group['sender'] + '→' + group['receiver']))

        # collect lr_means from matrix for these interactions
        records = []
        for interaction_key in sig_interactions:
            parts = interaction_key.split('|')
            if len(parts) < 4:
                continue
            sender, receiver, lig, rec = parts[0], parts[1], parts[2], parts[3]
            full_key = f"{sender}|{receiver}|{lig}|{rec}"
            if full_key not in mat.index:
                continue
            row = mat.loc[full_key]
            for animal, val in row.items():
                if pd.notna(val):
                    records.append({
                        'lr_pair': f"{lig}|{rec}",
                        'louvain_combo': f"{sender}→{receiver}",
                        'animal': animal,
                        'lr_means': float(val),
                        'age': age_row[animal],
                        'sex': sex_row[animal]
                    })

        if not records:
            continue

        df_long = pd.DataFrame(records)
        unique_lrs = sorted(df_long['lr_pair'].unique())
        n_lrs = len(unique_lrs)

        fig, axes = plt.subplots(1, n_lrs, figsize=(6 * n_lrs, 5),
                                  squeeze=False)
        axes = axes[0]

        louvain_str = ', '.join(louvain_pairs[:8])
        if len(louvain_pairs) > 8:
            louvain_str += f' +{len(louvain_pairs)-8} more'

        fig.suptitle(f"{region} — {combo_label}\nLouvain pairs: {louvain_str}",
                     fontsize=10, fontweight='bold')

        for ax, lr in zip(axes, unique_lrs):
            sub = df_long[df_long['lr_pair'] == lr]
            colors = sub['sex'].map({'M': '#1f77b4', 'F': '#e91e63'})

            ax.scatter(sub['age'], sub['lr_means'], c=colors, alpha=0.5, s=25, zorder=2)

            slope, intercept, r, p, se = stats.linregress(sub['age'], sub['lr_means'])
            x_range = np.linspace(sub['age'].min(), sub['age'].max(), 100)
            ax.plot(x_range, slope * x_range + intercept, 'k-', lw=2, zorder=3)

            n_lou = sub['louvain_combo'].nunique()
            ax.set_title(f"{lr}\nr={r:.2f} p={p:.2e}\n{n_lou} Louvain combos, {len(sub)} points",
                         fontsize=9)
            ax.set_xlabel("Age (years)", fontsize=9)
            ax.set_ylabel("lr_means", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(handles=[Patch(color='#e91e63', label='F'),
                                Patch(color='#1f77b4', label='M')],
                      fontsize=8, loc='upper right')

        plt.tight_layout()
        safe_combo = combo_label.replace('→', '_to_').replace('/', '_')
        fname = OUT / f"{region}_{safe_combo}_NLGN1_NRXN.png"
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {fname.name}")

print("\nDone.")
