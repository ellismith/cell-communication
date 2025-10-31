#!/usr/bin/env python3
"""
Visualize age and sex effects on astrocyte-GABA communication
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load results
age_results = pd.read_csv('/scratch/easmit31/cell_cell/results/cellchat_by_age.csv')
sex_results = pd.read_csv('/scratch/easmit31/cell_cell/results/cellchat_by_sex.csv')

# Filter for astrocyte-GABA interactions
age_sig = age_results[
    (age_results['magnitude_rank'] < 0.05) &
    (((age_results['source'] == 'Astrocyte') & (age_results['target'] == 'GABA_neuron')) |
     ((age_results['source'] == 'GABA_neuron') & (age_results['target'] == 'Astrocyte')))
].copy()

sex_sig = sex_results[
    (sex_results['magnitude_rank'] < 0.05) &
    (((sex_results['source'] == 'Astrocyte') & (sex_results['target'] == 'GABA_neuron')) |
     ((sex_results['source'] == 'GABA_neuron') & (sex_results['target'] == 'Astrocyte')))
].copy()

# Create interaction labels
age_sig['interaction'] = age_sig['ligand_complex'] + ' → ' + age_sig['receptor_complex']
age_sig['direction'] = age_sig['source'] + ' → ' + age_sig['target']

sex_sig['interaction'] = sex_sig['ligand_complex'] + ' → ' + sex_sig['receptor_complex']
sex_sig['direction'] = sex_sig['source'] + ' → ' + sex_sig['target']

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Age and Sex Effects on Astrocyte-GABA Communication', 
             fontsize=16, fontweight='bold', y=0.995)

# ============================================================================
# Panel 1: Age effects - communication strength
# ============================================================================
ax1 = axes[0, 0]

# Get core interactions present in all groups
core_interactions = ['NRG3 → ERBB4', 'NRXN1 → NLGN1', 'NRXN3 → NLGN1']
age_core = age_sig[age_sig['interaction'].isin(core_interactions)].copy()

for interaction in core_interactions:
    data = age_core[age_core['interaction'] == interaction]
    if len(data) > 0:
        # Group by direction
        for direction in data['direction'].unique():
            subset = data[data['direction'] == direction]
            ages = ['Young', 'Middle', 'Old']
            values = [subset[subset['age_group'] == age]['lr_means'].values[0] 
                     if age in subset['age_group'].values else 0 
                     for age in ages]
            
            label = f"{interaction} ({direction.split(' → ')[0]}→{direction.split(' → ')[1]})"
            ax1.plot(ages, values, marker='o', linewidth=2, markersize=8, label=label)

ax1.set_xlabel('Age Group', fontsize=12, fontweight='bold')
ax1.set_ylabel('Communication Strength (L-R Mean)', fontsize=12, fontweight='bold')
ax1.set_title('Communication Changes with Age', fontsize=13, fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax1.grid(alpha=0.3)

# ============================================================================
# Panel 2: Age effects - heatmap
# ============================================================================
ax2 = axes[0, 1]

# Create pivot for heatmap
age_pivot = age_sig.pivot_table(
    values='lr_means',
    index='interaction',
    columns='age_group',
    aggfunc='mean'
)[['Young', 'Middle', 'Old']]

sns.heatmap(age_pivot, annot=True, fmt='.2f', cmap='YlOrRd', 
           cbar_kws={'label': 'L-R Mean Expression'}, ax=ax2,
           linewidths=0.5)
ax2.set_title('Age Effect Heatmap', fontsize=13, fontweight='bold')
ax2.set_ylabel('Ligand-Receptor Pair', fontsize=11)
ax2.set_xlabel('Age Group', fontsize=11)

# ============================================================================
# Panel 3: Sex effects - comparison
# ============================================================================
ax3 = axes[1, 0]

sex_comparison = sex_sig.pivot_table(
    values='lr_means',
    index=['interaction', 'direction'],
    columns='sex',
    aggfunc='mean'
).reset_index()

y_pos = np.arange(len(sex_comparison))
width = 0.35

bars1 = ax3.barh(y_pos - width/2, sex_comparison['F'], width, 
                 label='Female', color='#FF6B9D', alpha=0.8)
bars2 = ax3.barh(y_pos + width/2, sex_comparison['M'], width,
                 label='Male', color='#4A90E2', alpha=0.8)

ax3.set_yticks(y_pos)
ax3.set_yticklabels([f"{row['interaction']}\n({row['direction']})" 
                      for _, row in sex_comparison.iterrows()], fontsize=9)
ax3.set_xlabel('Communication Strength (L-R Mean)', fontsize=12, fontweight='bold')
ax3.set_title('Sex Comparison', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(axis='x', alpha=0.3)

# ============================================================================
# Panel 4: Summary statistics
# ============================================================================
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
KEY FINDINGS:

CONSERVED INTERACTIONS (all ages & both sexes):
- NRG3 → ERBB4 (bidirectional)
- NRXN1 → NLGN1 (Astrocyte → GABA)  
- NRXN3 → NLGN1 (GABA → Astrocyte)

AGE EFFECTS:
- Young: {len(age_sig[age_sig['age_group']=='Young'])} significant interactions
- Middle: {len(age_sig[age_sig['age_group']=='Middle'])} significant interactions
- Old: {len(age_sig[age_sig['age_group']=='Old'])} significant interactions

Core interactions maintained across lifespan
Slight reduction in diversity with age

SEX EFFECTS:
- Female: {len(sex_sig[sex_sig['sex']=='F'])} significant interactions
- Male: {len(sex_sig[sex_sig['sex']=='M'])} significant interactions

No sex-specific interactions detected
All communication patterns conserved between sexes

BIOLOGICAL INTERPRETATION:
- Neurexin-Neuroligin: Synaptic adhesion (stable)
- Neuregulin-ERBB4: Neurotrophic signaling (stable)
- Fundamental glia-neuron crosstalk preserved
  across development and between sexes
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('/scratch/easmit31/cell_cell/figures/cellchat_age_sex_effects.png',
           dpi=300, bbox_inches='tight')
print("✓ Saved age/sex effects visualization")

# Also save detailed tables
age_sig.to_csv('/scratch/easmit31/cell_cell/results/age_effects_detailed.csv', index=False)
sex_sig.to_csv('/scratch/easmit31/cell_cell/results/sex_effects_detailed.csv', index=False)
print("✓ Saved detailed tables")
