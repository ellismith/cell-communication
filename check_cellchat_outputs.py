#!/usr/bin/env python3
import pandas as pd
import glob

# Get one specific result file
result_files = glob.glob("/scratch/easmit31/cell_cell/results/pairwise/*_results.csv")
print(f"Found {len(result_files)} result files\n")

# Look at a few specific files
test_files = [f for f in result_files if 'Glutamatergic' in f and 'GABA' in f]
print(f"Glutamatergic-GABA files: {len(test_files)}")
if test_files:
    print(f"Example: {test_files[0]}\n")
    
    df = pd.read_csv(test_files[0])
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}\n")
    
    print("First 20 rows:")
    print(df[['source', 'target', 'ligand_complex', 'receptor_complex', 'lr_means', 'qval']].head(20))
    
    print(f"\n\nUnique source regions: {df['source'].str.rsplit('_', n=1).str[-1].unique()}")
    print(f"Unique target regions: {df['target'].str.rsplit('_', n=1).str[-1].unique()}")
    
    # Check if lr_means varies by region for the same ligand-receptor pair
    print(f"\n\nChecking NRXN1-NLGN1 across different region pairs:")
    nrxn1 = df[(df['ligand_complex'] == 'NRXN1') & (df['receptor_complex'] == 'NLGN1')]
    print(nrxn1[['source', 'target', 'lr_means', 'qval']])
    
    print(f"\n\nDoes lr_means vary for NRXN1-NLGN1?")
    print(f"  Min: {nrxn1['lr_means'].min():.6f}")
    print(f"  Max: {nrxn1['lr_means'].max():.6f}")
    print(f"  All same? {nrxn1['lr_means'].nunique() == 1}")

