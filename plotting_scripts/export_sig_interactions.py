#!/usr/bin/env python3
"""
Export significant age-associated interactions for a given region and category to CSV.

Usage:
    python export_sig_interactions.py --region dlPFC --category "Synaptic adhesion"
    python export_sig_interactions.py --region HIP --category "Glutamate signaling" --q_thresh 0.1
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--region', required=True)
parser.add_argument('--category', required=True)
parser.add_argument('--q_thresh', type=float, default=0.05)
args = parser.parse_args()

BASE_DIR = Path("/scratch/easmit31/cell_cell/results/within_region_analysis_corrected")
ANN_FILE = Path("/scratch/easmit31/cell_cell/cpdb_lr_annotations.csv")
OUT_DIR  = BASE_DIR / "sig_interaction_tables"
OUT_DIR.mkdir(exist_ok=True)

ann = pd.read_csv(ANN_FILE)
ann['lr_pair'] = ann['ligand'] + '|' + ann['receptor']
cat_pairs = set(ann[ann['broad_category'] == args.category]['lr_pair'])
print(f"'{args.category}' LR pairs: {len(cat_pairs)}")

csv = BASE_DIR / "regression_results" / f"regression_{args.region}" / f"whole_{args.region.lower()}_age_sex_regression.csv"
print(f"Loading {csv}...")

chunks = []
for chunk in pd.read_csv(csv, chunksize=500000):
    parts = chunk['interaction'].str.split('|', expand=True)
    chunk['sender']   = parts[0]
    chunk['receiver'] = parts[1]
    chunk['ligand']   = parts[2]
    chunk['receptor'] = parts[3]
    chunk['lr_pair']  = chunk['ligand'] + '|' + chunk['receptor']
    chunk['sender_broad']   = chunk['sender'].str.replace(r'_\d+$', '', regex=True)
    chunk['receiver_broad'] = chunk['receiver'].str.replace(r'_\d+$', '', regex=True)
    sub = chunk[(chunk['age_qval'] < args.q_thresh) & (chunk['lr_pair'].isin(cat_pairs))]
    if len(sub):
        chunks.append(sub)

if not chunks:
    print("No significant interactions found.")
    exit(0)

sig = pd.concat(chunks)
sig = sig.merge(ann[['lr_pair','broad_category','classification']].drop_duplicates(),
                on='lr_pair', how='left')

cols = ['interaction','sender','sender_broad','receiver','receiver_broad',
        'ligand','receptor','lr_pair','broad_category','classification',
        'n_animals','mean_lr_means','age_coef','age_stderr','age_pval','age_qval',
        'sex_coef','sex_stderr','sex_pval','sex_qval','r_squared']
cols = [c for c in cols if c in sig.columns]
sig = sig[cols].sort_values('age_qval')

cat_slug = args.category.replace(' ','_').replace('/','_')
q_tag = f"_q{str(args.q_thresh).replace('.','p')}"
out = OUT_DIR / f"{args.region}_{cat_slug}{q_tag}.csv"
sig.to_csv(out, index=False)
print(f"Saved {len(sig)} interactions to {out}")
print(sig[['sender_broad','receiver_broad','lr_pair','age_coef','age_qval']].head(20).to_string())
