#!/usr/bin/env python3
"""
Compare CellChatDB vs CellPhoneDB v5 results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*70)
print("COMPARING OLD vs NEW DATABASE RESULTS")
print("="*70)

# Directories
old_dir = "results/u01_per_individual"
new_dir = "results/u01_per_individual_CPDB5"

# Get list of individuals
old_files = [f for f in os.listdir(old_dir) if f.endswith('_results.csv')]
new_files = [f for f in os.listdir(new_dir) if f.endswith('_results_CPDB5.csv')]

old_ids = set([f.replace('_results.csv', '') for f in old_files])
new_ids = set([f.replace('_results_CPDB5.csv', '') for f in new_files])

common_ids = old_ids & new_ids

print(f"\nIndividuals in old results: {len(old_ids)}")
print(f"Individuals in new results: {len(new_ids)}")
print(f"Common individuals: {len(common_ids)}")

if len(common_ids) == 0:
    print("\nERROR: No common individuals found!")
    exit(1)

# Compare for each individual
comparison_data = []

for individual in sorted(common_ids):
    old_file = f"{old_dir}/{individual}_results.csv"
    new_file = f"{new_dir}/{individual}_results_CPDB5.csv"
    
    old_df = pd.read_csv(old_file)
    new_df = pd.read_csv(new_file)
    
    # Count stats
    old_total = len(old_df)
    new_total = len(new_df)
    old_sig = (old_df['qval'] < 0.05).sum()
    new_sig = (new_df['qval'] < 0.05).sum()
    
    # Unique L-R pairs
    old_pairs = set(old_df['ligand_complex'] + ' -> ' + old_df['receptor_complex'])
    new_pairs = set(new_df['ligand_complex'] + ' -> ' + new_df['receptor_complex'])
    
    overlap_pairs = old_pairs & new_pairs
    only_old = old_pairs - new_pairs
    only_new = new_pairs - old_pairs
    
    comparison_data.append({
        'individual': individual,
        'age': old_df['age'].iloc[0],
        'sex': old_df['sex'].iloc[0],
        'old_total': old_total,
        'new_total': new_total,
        'old_sig': old_sig,
        'new_sig': new_sig,
        'old_unique_pairs': len(old_pairs),
        'new_unique_pairs': len(new_pairs),
        'overlap_pairs': len(overlap_pairs),
        'only_old_pairs': len(only_old),
        'only_new_pairs': len(only_new)
    })

comparison_df = pd.DataFrame(comparison_data)

# Summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\nTotal interactions (mean ± std):")
print(f"  Old (CellChatDB):      {comparison_df['old_total'].mean():,.0f} ± {comparison_df['old_total'].std():,.0f}")
print(f"  New (CellPhoneDB v5):  {comparison_df['new_total'].mean():,.0f} ± {comparison_df['new_total'].std():,.0f}")
print(f"  Difference:            {(comparison_df['new_total'] - comparison_df['old_total']).mean():,.0f}")

print(f"\nSignificant interactions (mean ± std):")
print(f"  Old (CellChatDB):      {comparison_df['old_sig'].mean():,.0f} ± {comparison_df['old_sig'].std():,.0f}")
print(f"  New (CellPhoneDB v5):  {comparison_df['new_sig'].mean():,.0f} ± {comparison_df['new_sig'].std():,.0f}")
print(f"  Difference:            {(comparison_df['new_sig'] - comparison_df['old_sig']).mean():,.0f}")

print(f"\nUnique L-R pairs (mean ± std):")
print(f"  Old (CellChatDB):      {comparison_df['old_unique_pairs'].mean():,.0f} ± {comparison_df['old_unique_pairs'].std():,.0f}")
print(f"  New (CellPhoneDB v5):  {comparison_df['new_unique_pairs'].mean():,.0f} ± {comparison_df['new_unique_pairs'].std():,.0f}")
print(f"  Overlap:               {comparison_df['overlap_pairs'].mean():,.0f} ± {comparison_df['overlap_pairs'].std():,.0f}")
print(f"  Only in old:           {comparison_df['only_old_pairs'].mean():,.0f} ± {comparison_df['only_old_pairs'].std():,.0f}")
print(f"  Only in new:           {comparison_df['only_new_pairs'].mean():,.0f} ± {comparison_df['only_new_pairs'].std():,.0f}")

# Save comparison
comparison_df.to_csv("results/database_comparison_summary.csv", index=False)
print(f"\n✓ Saved comparison to results/database_comparison_summary.csv")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Total interactions
ax = axes[0, 0]
x = np.arange(len(comparison_df))
width = 0.35
ax.bar(x - width/2, comparison_df['old_total'], width, label='CellChatDB', alpha=0.8)
ax.bar(x + width/2, comparison_df['new_total'], width, label='CellPhoneDB v5', alpha=0.8)
ax.set_xlabel('Individual')
ax.set_ylabel('Total Interactions')
ax.set_title('Total Interactions per Individual')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Significant interactions
ax = axes[0, 1]
ax.bar(x - width/2, comparison_df['old_sig'], width, label='CellChatDB', alpha=0.8)
ax.bar(x + width/2, comparison_df['new_sig'], width, label='CellPhoneDB v5', alpha=0.8)
ax.set_xlabel('Individual')
ax.set_ylabel('Significant Interactions (q<0.05)')
ax.set_title('Significant Interactions per Individual')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Scatter comparison
ax = axes[1, 0]
ax.scatter(comparison_df['old_sig'], comparison_df['new_sig'], alpha=0.6, s=100)
ax.plot([0, comparison_df['old_sig'].max()], [0, comparison_df['old_sig'].max()], 
        'r--', alpha=0.5, label='y=x')
ax.set_xlabel('Significant in CellChatDB')
ax.set_ylabel('Significant in CellPhoneDB v5')
ax.set_title('Correlation of Significant Interactions')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: L-R pair overlap
ax = axes[1, 1]
overlap_means = [
    comparison_df['overlap_pairs'].mean(),
    comparison_df['only_old_pairs'].mean(),
    comparison_df['only_new_pairs'].mean()
]
labels = ['Overlap', 'Only CellChatDB', 'Only CellPhoneDB v5']
colors = ['green', 'blue', 'red']
ax.bar(labels, overlap_means, color=colors, alpha=0.7)
ax.set_ylabel('Mean # of L-R Pairs')
ax.set_title('L-R Pair Overlap Between Databases')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/database_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved plots to results/database_comparison.png")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("CellPhoneDB v5 provides:")
print(f"  • {(comparison_df['new_total'] - comparison_df['old_total']).mean():,.0f} more interactions per individual")
print(f"  • {(comparison_df['new_sig'] - comparison_df['old_sig']).mean():,.0f} more significant interactions per individual")
print(f"  • {comparison_df['only_new_pairs'].mean():,.0f} unique L-R pairs not in CellChatDB")
print("\nRecommendation: Use CellPhoneDB v5 results for all downstream analyses!")
