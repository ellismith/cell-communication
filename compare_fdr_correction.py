#!/usr/bin/env python3
"""
Compare results with and without FDR correction
"""
import pandas as pd
from statsmodels.stats.multitest import multipletests

# Pick one example result file
result_file = "/scratch/easmit31/cell_cell/results/pairwise/Astrocyte_GABA_results.csv"

print("="*70)
print("Comparing FDR Correction Impact")
print("="*70)

# Load existing results
df = pd.read_csv(result_file)

print(f"\n1. ORIGINAL DATA:")
print(f"   Total interactions tested: {len(df):,}")
print(f"   Columns: {df.columns.tolist()}")

# Check if qval already exists
if 'qval' in df.columns:
    print("\n   ⚠️  qval column already exists in file")
    qval_exists = True
else:
    print("\n   ℹ️  No qval column - will calculate it")
    qval_exists = False
    # Calculate FDR correction
    _, df['qval'], _, _ = multipletests(df['magnitude_rank'], method='fdr_bh')

print(f"\n2. SIGNIFICANCE COMPARISON:")
print(f"   Using raw p-value (magnitude_rank < 0.05):")
sig_raw = df[df['magnitude_rank'] < 0.05]
print(f"      Significant: {len(sig_raw):,} ({100*len(sig_raw)/len(df):.1f}%)")

print(f"\n   Using FDR-corrected q-value (qval < 0.05):")
sig_fdr = df[df['qval'] < 0.05]
print(f"      Significant: {len(sig_fdr):,} ({100*len(sig_fdr)/len(df):.1f}%)")

print(f"\n   Difference: {len(sig_raw) - len(sig_fdr):,} interactions removed by FDR correction")

print(f"\n3. WHAT GOT FILTERED OUT:")
# Interactions significant by raw p but not by FDR
filtered_out = df[(df['magnitude_rank'] < 0.05) & (df['qval'] >= 0.05)]
print(f"   {len(filtered_out):,} interactions were significant (p<0.05) but NOT after FDR (q>=0.05)")

if len(filtered_out) > 0:
    print(f"\n   Top 10 filtered interactions (borderline significance):")
    print(filtered_out.nsmallest(10, 'magnitude_rank')[['source', 'target', 'ligand_complex', 'receptor_complex', 'lr_means', 'magnitude_rank', 'qval']])

print(f"\n4. TOP INTERACTIONS (should be same for both):")
print(f"   Top 10 by FDR-corrected q-value:")
print(sig_fdr.nsmallest(10, 'qval')[['source', 'target', 'ligand_complex', 'receptor_complex', 'lr_means', 'magnitude_rank', 'qval']])

print(f"\n5. RECOMMENDATION:")
if len(sig_raw) - len(sig_fdr) > 100:
    print(f"   ⚠️  FDR removes {len(sig_raw) - len(sig_fdr):,} interactions - this is appropriate for multiple testing")
    print(f"   ✓ Use FDR-corrected results (qval < 0.05) for publication")
elif len(sig_raw) - len(sig_fdr) < 50:
    print(f"   ✓ FDR correction is conservative but only removes {len(sig_raw) - len(sig_fdr):,} interactions")
    print(f"   ✓ Both approaches are reasonable, but FDR is more rigorous")
else:
    print(f"   ℹ️  FDR removes a moderate number ({len(sig_raw) - len(sig_fdr):,}) of interactions")
    print(f"   ✓ Use FDR-corrected results for more confidence")

print("\n" + "="*70)
