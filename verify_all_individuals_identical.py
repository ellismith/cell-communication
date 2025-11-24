#!/usr/bin/env python3
"""
Verify that old and new results are identical for ALL 55 individuals
"""
import pandas as pd
import os

old_dir = "results/u01_per_individual"
new_dir = "results/u01_per_individual_CPDB5"

# Get all individuals
old_files = [f for f in os.listdir(old_dir) if f.endswith('_results.csv')]
new_files = [f for f in os.listdir(new_dir) if f.endswith('_results_CPDB5.csv')]

old_ids = sorted([f.replace('_results.csv', '') for f in old_files])
new_ids = sorted([f.replace('_results_CPDB5.csv', '') for f in new_files])

print("="*70)
print("VERIFYING ALL 55 INDIVIDUALS")
print("="*70)
print(f"\nOld results: {len(old_ids)} individuals")
print(f"New results: {len(new_ids)} individuals")

if old_ids != new_ids:
    print("\n⚠️ WARNING: Different individuals in each dataset!")
    print(f"Missing from new: {set(old_ids) - set(new_ids)}")
    print(f"Extra in new: {set(new_ids) - set(old_ids)}")
    common_ids = sorted(set(old_ids) & set(new_ids))
else:
    common_ids = old_ids
    print(f"✓ Same {len(common_ids)} individuals in both")

# Check each individual
all_identical = True
differences = []

for idx, individual in enumerate(common_ids, 1):
    old_file = f"{old_dir}/{individual}_results.csv"
    new_file = f"{new_dir}/{individual}_results_CPDB5.csv"
    
    old_df = pd.read_csv(old_file)
    new_df = pd.read_csv(new_file)
    
    # Create comparison key
    old_df['key'] = (old_df['source'] + '|' + old_df['target'] + '|' + 
                     old_df['ligand_complex'] + '|' + old_df['receptor_complex'])
    new_df['key'] = (new_df['source'] + '|' + new_df['target'] + '|' + 
                     new_df['ligand_complex'] + '|' + new_df['receptor_complex'])
    
    # Compare
    same_rows = len(old_df) == len(new_df)
    same_keys = set(old_df['key']) == set(new_df['key'])
    
    if same_rows and same_keys:
        # Check if values are identical
        merged = old_df[['key', 'lr_means', 'qval']].merge(
            new_df[['key', 'lr_means', 'qval']], 
            on='key', suffixes=('_old', '_new')
        )
        
        same_lr_means = (merged['lr_means_old'] == merged['lr_means_new']).all()
        same_qval = (merged['qval_old'] == merged['qval_new']).all()
        
        if same_lr_means and same_qval:
            status = "✓ IDENTICAL"
        else:
            status = "✗ VALUES DIFFER"
            all_identical = False
            differences.append({
                'individual': individual,
                'issue': 'Different values',
                'n_lr_means_diff': (~(merged['lr_means_old'] == merged['lr_means_new'])).sum(),
                'n_qval_diff': (~(merged['qval_old'] == merged['qval_new'])).sum()
            })
    else:
        status = "✗ DIFFERENT"
        all_identical = False
        differences.append({
            'individual': individual,
            'issue': 'Different interactions',
            'old_rows': len(old_df),
            'new_rows': len(new_df),
            'old_unique': len(set(old_df['key'])),
            'new_unique': len(set(new_df['key']))
        })
    
    if idx % 10 == 0 or not (same_rows and same_keys and same_lr_means and same_qval):
        print(f"[{idx:2d}/{len(common_ids)}] {individual}: {status}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if all_identical:
    print("✓ ALL 55 INDIVIDUALS HAVE IDENTICAL RESULTS")
    print("\nConclusion: CellChatDB and CellPhoneDB v5 give the same results")
    print("for your dataset. You can use either - no need to redo anything!")
else:
    print(f"✗ Found differences in {len(differences)} individuals")
    print("\nDifferences:")
    for diff in differences:
        print(f"\n{diff['individual']}:")
        for key, val in diff.items():
            if key != 'individual':
                print(f"  {key}: {val}")

