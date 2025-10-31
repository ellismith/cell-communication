#!/usr/bin/env python3
"""
Comprehensive visualization of astrocyte-GABA communication with arrows
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load results
results = pd.read_csv("/scratch/easmit31/cell_cell/results/cellchat_significant_interactions.csv")

# Filter for astrocyte-GABA interactions
astro_gaba = results[
    ((results['source'] == 'Astrocyte') & (results['target'] == 'GABA_neuron')) |
    ((results['source'] == 'GABA_neuron') & (results['target'] == 'Astrocyte'))
].copy()

print(f"Total astrocyte-GABA interactions: {len(astro_gaba)}")

# Create interaction labels
astro_gaba['interaction'] = astro_gaba['ligand_complex'] + ' → ' + astro_gaba['receptor_complex']
astro_gaba['direction_label'] = astro_gaba.apply(
    lambda x: 'Astrocyte → GABA' if x['source'] == 'Astrocyte' else 'GABA → Astrocyte', 
    axis=1
)

# Sort by significance
astro_gaba = astro_gaba.sort_values('magnitude_rank')

# Create figure with multiple panels
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# ============================================================================
# Panel 1: Communication strength with directional arrows
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])

# Separate by direction
astro_to_gaba = astro_gaba[astro_gaba['source'] == 'Astrocyte'].sort_values('lr_means', ascending=False)
gaba_to_astro = astro_gaba[astro_gaba['source'] == 'GABA_neuron'].sort_values('lr_means', ascending=False)

y_pos = np.arange(len(astro_gaba))
colors = ['#FF6B6B' if x == 'Astrocyte' else '#4ECDC4' 
          for x in astro_gaba['source']]

bars = ax1.barh(y_pos, astro_gaba['lr_means'], color=colors, alpha=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(astro_gaba['interaction'], fontsize=10)
ax1.set_xlabel('L-R Mean Expression (Communication Strength)', fontsize=12)
ax1.set_title('All Significant Astrocyte ↔ GABA Interactions', fontsize=14, fontweight='bold')

# Add arrows to show directionality
for i, (idx, row) in enumerate(astro_gaba.iterrows()):
    if row['source'] == 'Astrocyte':
        ax1.text(row['lr_means'] + 0.05, i, '→', fontsize=16, va='center', color='#FF6B6B')
    else:
        ax1.text(row['lr_means'] + 0.05, i, '←', fontsize=16, va='center', color='#4ECDC4')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF6B6B', alpha=0.7, label='Astrocyte → GABA'),
    Patch(facecolor='#4ECDC4', alpha=0.7, label='GABA → Astrocyte')
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=11)
ax1.grid(axis='x', alpha=0.3)

# ============================================================================
# Panel 2: Chord-like diagram showing flow
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])

# Aggregate by direction
direction_counts = astro_gaba.groupby('direction_label').agg({
    'lr_means': 'mean',
    'interaction': 'count'
}).reset_index()
direction_counts.columns = ['Direction', 'Avg_Strength', 'Count']

# Create arrow plot
y_positions = [0, 1]
for i, row in direction_counts.iterrows():
    if 'Astrocyte → GABA' in row['Direction']:
        ax2.arrow(0.2, y_positions[0], 0.6, 0, head_width=0.15, head_length=0.1, 
                 fc='#FF6B6B', ec='#FF6B6B', linewidth=3, alpha=0.7)
        ax2.text(0.5, y_positions[0] + 0.25, f"{row['Count']} interactions\nAvg strength: {row['Avg_Strength']:.2f}",
                ha='center', fontsize=11, fontweight='bold')
    else:
        ax2.arrow(0.8, y_positions[1], -0.6, 0, head_width=0.15, head_length=0.1,
                 fc='#4ECDC4', ec='#4ECDC4', linewidth=3, alpha=0.7)
        ax2.text(0.5, y_positions[1] - 0.25, f"{row['Count']} interactions\nAvg strength: {row['Avg_Strength']:.2f}",
                ha='center', fontsize=11, fontweight='bold')

# Cell type labels
ax2.text(0.1, y_positions[0], 'Astrocyte', fontsize=14, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='#FF6B6B', alpha=0.3))
ax2.text(0.1, y_positions[1], 'GABA', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#4ECDC4', alpha=0.3))
ax2.text(0.85, y_positions[0], 'GABA', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#4ECDC4', alpha=0.3))
ax2.text(0.85, y_positions[1], 'Astrocyte', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FF6B6B', alpha=0.3))

ax2.set_xlim(-0.1, 1.1)
ax2.set_ylim(-0.5, 1.5)
ax2.axis('off')
ax2.set_title('Bidirectional Communication Flow', fontsize=14, fontweight='bold')

# ============================================================================
# Panel 3: Significance ranking
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])

bars = ax3.barh(y_pos, -np.log10(astro_gaba['magnitude_rank'] + 1e-10), color=colors, alpha=0.7)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(astro_gaba['interaction'], fontsize=10)
ax3.set_xlabel('-log10(Magnitude Rank)', fontsize=12)
ax3.set_title('Statistical Significance', fontsize=14, fontweight='bold')
ax3.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=2, alpha=0.5, label='p=0.05')
ax3.legend(fontsize=10)
ax3.grid(axis='x', alpha=0.3)

# ============================================================================
# Panel 4: Heatmap of L-R pairs
# ============================================================================
ax4 = fig.add_subplot(gs[2, :])

# Create matrix for heatmap
pivot_data = astro_gaba.pivot_table(
    values='lr_means',
    index='interaction',
    columns='direction_label',
    fill_value=0
)

sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', 
           cbar_kws={'label': 'L-R Mean Expression'}, ax=ax4,
           linewidths=0.5, linecolor='gray')
ax4.set_title('Communication Strength Matrix', fontsize=14, fontweight='bold')
ax4.set_xlabel('Direction', fontsize=12)
ax4.set_ylabel('Ligand-Receptor Pair', fontsize=12)

plt.savefig('/scratch/easmit31/cell_cell/figures/cellchat_comprehensive.png', 
           dpi=300, bbox_inches='tight')
print("\n✓ Saved comprehensive visualization")

# ============================================================================
# Create summary table
# ============================================================================
summary = astro_gaba[['source', 'target', 'ligand_complex', 'receptor_complex', 
                      'lr_means', 'magnitude_rank']].copy()
summary = summary.sort_values('magnitude_rank')
summary.to_csv('/scratch/easmit31/cell_cell/results/cellchat_summary_table.csv', index=False)

print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(summary.to_string(index=False))
print("\n✓ Saved summary table")

