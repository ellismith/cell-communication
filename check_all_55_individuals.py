#!/usr/bin/env python3
"""
Check ALL 55 individuals systematically
"""
import pandas as pd
import numpy as np
import os

old_dir = "results/u01_per_individual"
new_dir = "results/u01_per_individual_CPDB5"

# Get all individuals
old_files = sorted([f for f in os.listdir(old_dir) if f.endswith('_results.csv')])
new_files = sorted([f for f in os.listdir(new_dir) if f.endswith('_results_CPDB5.csv')])

old_ids = [f.replace('_results.csv', '') for f in old_files]
new_ids = [f.replace('_results_CPDB5.csv', '') for f in new_files]

print("="*70)
print(f"CHECKING ALL {len(old_ids)} INDIVIDUALS")
print("="*70)

if set(old_ids) != set(new_ids):
    print("⚠️  Different individuals!")
    print(f"Only in old: {set(old_ids) - set(new_ids)}")
    print(f"Only in new: {set(new_ids) - set(old_ids)}")
    common = sorted(set(old_ids) & set(new_ids))
else:
    common = old_ids
    print(f"✓ Same {len(common)} individuals\n")

# Check each one
all_match = True
mismatches = []

for idx, individual in enumerate(common, 1):
    old_file = f"{old_dir}/{individual}_results.csv"
    new_file = f"{new_dir}/{individual}_results_CPDB5.csv"
    
    old = pd.read_csv(old_file)
    new = pd.read_csv(new_file)
    
    # Sort both
    old_sorted = old.sort_values(['source', 'target', 'ligand_complex', 'receptor_complex']).reset_index(drop=True)
    new_sorted = new.sort_values(['source', 'target', 'ligand_complex', 'receptor_complex']).reset_index(drop=True)
    
    # Check key columns
    same_shape = old_sorted.shape[0] == new_sorted.shape[0]
    same_source = (old_sorted['source'] == new_sorted['source']).all()
    same_target = (old_sorted['target'] == new_sorted['target']).all()
    same_ligand = (old_sorted['ligand_complex'] == new_sorted['ligand_complex']).all()
    same_receptor = (old_sorted['receptor_complex'] == new_sorted['receptor_complex']).all()
    same_lr_means = np.allclose(old_sorted['lr_means'], new_sorted['lr_means'], equal_nan=True)
    same_qval = np.allclose(old_sorted['qval'], new_sorted['qval'], equal_nan=True)
    
    all_checks_pass = (same_shape and same_source and same_target and 
                       same_ligand and same_receptor and same_lr_means and same_qval)
    
    if all_checks_pass:
        status = "✓"
    else:
        status = "✗"
        all_match = False
        mismatches.append({
            'individual': individual,
            'same_shape': same_shape,
            'same_source': same_source,
            'same_target': same_target,
            'same_ligand': same_ligand,
            'same_receptor': same_receptor,
            'same_lr_means': same_lr_means,
            'same_qval': same_qval,
            'old_rows': len(old),
            'new_rows': len(new)
        })
    
    if idx % 5 == 0 or not all_checks_pass:
        print(f"[{idx:2d}/{len(common)}] {individual:8s} {status}")

print("\n" + "="*70)
print("FINAL RESULT")
print("="*70)

if all_match:
    print(f"✓✓✓ ALL {len(common)} INDIVIDUALS HAVE IDENTICAL DATA")
    print("\nAfter sorting by interaction, all values match:")
    print("  - source/target")
    print("  - ligand/receptor")
    print("  - lr_means")
    print("  - qval")
    print("\nConclusion: The results are IDENTICAL.")
    print("Row order differs but data is the same.")
else:
    print(f"✗ FOUND {len(mismatches)} INDIVIDUALS WITH DIFFERENCES\n")
    for m in mismatches:
        print(f"\n{m['individual']}:")
        for k, v in m.items():
            if k != 'individual':
                print(f"  {k}: {v}")

