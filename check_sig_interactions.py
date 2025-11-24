#!/usr/bin/env python3
import pandas as pd
import glob

result_files = glob.glob("/scratch/easmit31/cell_cell/results/pairwise/*_results.csv")

total_interactions = 0
sig_raw = 0
sig_fdr = 0

for f in result_files:
    df = pd.read_csv(f)
    total_interactions += len(df)
    sig_raw += (df['magnitude_rank'] < 0.05).sum()
    sig_fdr += (df['qval'] < 0.05).sum()

print(f"Total interactions tested: {total_interactions:,}")
print(f"Significant (raw p<0.05): {sig_raw:,} ({100*sig_raw/total_interactions:.1f}%)")
print(f"Significant (FDR q<0.05): {sig_fdr:,} ({100*sig_fdr/total_interactions:.1f}%)")
print(f"\nFDR removed: {sig_raw - sig_fdr:,} interactions")
