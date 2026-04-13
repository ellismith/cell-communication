#!/usr/bin/env python3
"""
Build lr_means matrix for a region: rows=interactions, columns=animal_ids.
Age and sex prepended as first two rows. Missing interactions get NaN.
Usage:
    python build_lr_matrix.py --region HIP --threshold 2.0 --min_age 1.0
    python build_lr_matrix.py --region dlPFC --threshold 0 --min_age 1.0 --input_dir /path/to/files --output_name dlPFC_nothresh_minage1p0_matrix_attempt2.csv
"""
import pandas as pd
import numpy as np
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--region', type=str, required=True)
parser.add_argument('--threshold', type=float, default=2.0)
parser.add_argument('--min_age', type=float, default=1.0)
parser.add_argument('--input_dir', type=str, default='/scratch/easmit31/cell_cell/results/per_animal_louvain_corrected/')
parser.add_argument('--output_name', type=str, default=None)
args = parser.parse_args()

input_dir = args.input_dir
output_dir = '/scratch/easmit31/cell_cell/results/lr_matrices_corrected/'
os.makedirs(output_dir, exist_ok=True)

files = glob.glob(os.path.join(input_dir, f'*_{args.region}_louvain_results.csv'))
print(f"Found {len(files)} files for region {args.region}")
print(f"Input dir: {input_dir}")

animal_dfs = {}
for fpath in sorted(files):
    animal_id = os.path.basename(fpath).split(f'_{args.region}')[0]
    df = pd.read_csv(fpath, usecols=['source', 'target', 'ligand_complex', 'receptor_complex',
                                      'lr_means', 'age', 'sex'])
    df = df[(df['lr_means'] >= args.threshold) & (df['age'] >= args.min_age)]
    if df.empty:
        print(f"  {animal_id}: no rows above threshold, skipping")
        continue
    age = df['age'].iloc[0]
    sex = df['sex'].iloc[0]
    df['interaction'] = (df['source'] + '|' + df['target'] + '|' +
                         df['ligand_complex'] + '|' + df['receptor_complex'])
    series = df.groupby('interaction')['lr_means'].mean()
    series.name = animal_id
    animal_dfs[animal_id] = (age, sex, series)
    print(f"  {animal_id}: {len(series)} interactions above threshold")

if not animal_dfs:
    print("No data found, exiting.")
    exit(1)

matrix = pd.concat([v[2] for v in animal_dfs.values()], axis=1)
age_row = pd.Series({aid: v[0] for aid, v in animal_dfs.items()}, name='age')
sex_row = pd.Series({aid: v[1] for aid, v in animal_dfs.items()}, name='sex')
out = pd.concat([age_row.to_frame().T, sex_row.to_frame().T, matrix])

if args.output_name:
    output_file = os.path.join(output_dir, args.output_name)
else:
    output_file = os.path.join(output_dir, f'{args.region}_nothresh_minage{str(args.min_age).replace(".","p")}_matrix.csv')

out.to_csv(output_file)
print(f"\nSaved: {output_file}")
print(f"Shape: {matrix.shape[0]} interactions x {matrix.shape[1]} animals")
print(f"NaN rate: {matrix.isna().mean().mean():.2%}")
