#!/usr/bin/env python3
"""
Visualize CellChat results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results = pd.read_csv("/scratch/easmit31/cell_cell/results/cellchat_significant_interactions.csv")

# Filter for astrocyte-GABA interactions only (not GABA-GABA)
astro_gaba = results[
    ((results['source'] == 'Astrocyte') & (results['target'] == 'GABA_neuron')) |
    ((results['source'] == 'GABA_neuron') & (results['target'] == 'Astrocyte'))
].copy()

# Create interaction label
astro_gaba['interaction'] = (astro_gaba['ligand_complex'] + ' → ' + 
                              astro_gaba['receptor_complex'])
astro_gaba['direction'] = astro_gaba['source'] + ' → ' + astro_gaba['target']

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Communication strength by direction
sns.barplot(data=astro_gaba, x='direction', y='lr_means', 
            hue='interaction', ax=axes[0])
axes[0].set_title('Communication Strength by Direction')
axes[0].set_ylabel('L-R Mean Expression')
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 2: Magnitude rank (lower = more significant)
sns.barplot(data=astro_gaba.sort_values('magnitude_rank'), 
            x='interaction', y='magnitude_rank', 
            hue='direction', ax=axes[1])
axes[1].set_title('Significance of Interactions')
axes[1].set_ylabel('Magnitude Rank (lower = better)')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='Direction')

plt.tight_layout()
plt.savefig('/scratch/easmit31/cell_cell/figures/cellchat_overview.png', 
            dpi=300, bbox_inches='tight')
print("✓ Saved overview plot")

# Summary table
print("\n" + "="*70)
print("SUMMARY OF ASTROCYTE-GABA COMMUNICATION")
print("="*70)
print(astro_gaba[['source', 'target', 'ligand_complex', 'receptor_complex', 
                   'lr_means', 'magnitude_rank']].to_string(index=False))
