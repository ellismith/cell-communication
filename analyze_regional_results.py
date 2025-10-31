#!/usr/bin/env python3
"""
Analyze regional differences - with directional balance
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load regional results
regional = pd.read_csv('/scratch/easmit31/cell_cell/results/cellchat_by_region_all9.csv')

# Filter for astrocyte-GABA interactions
regional_sig = regional[
    (regional['magnitude_rank'] < 0.05) &
    (((regional['source'] == 'Astrocyte') & (regional['target'] == 'GABA_neuron')) |
     ((regional['source'] == 'GABA_neuron') & (regional['target'] == 'Astrocyte')))
].copy()

regional_sig['interaction'] = regional_sig['ligand_complex'] + ' → ' + regional_sig['receptor_complex']
regional_sig['direction'] = regional_sig['source'] + ' → ' + regional_sig['target']

print("="*70)
print("REGIONAL ANALYSIS - DIRECTIONAL SUMMARY")
print("="*70)

# Directional analysis per region
regions_list = sorted(regional_sig['region'].unique())
for region in regions_list:
    reg_data = regional_sig[regional_sig['region'] == region]
    a_to_g = reg_data[reg_data['source'] == 'Astrocyte']
    g_to_a = reg_data[reg_data['source'] == 'GABA_neuron']
    
    print(f"\n{region}:")
    print(f"  Astrocyte → GABA: {len(a_to_g)} interactions (avg strength: {a_to_g['lr_means'].mean():.2f})")
    print(f"  GABA → Astrocyte: {len(g_to_a)} interactions (avg strength: {g_to_a['lr_means'].mean():.2f})")

# Create visualization
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
fig.suptitle('Regional Differences in Astrocyte-GABA Communication', 
             fontsize=16, fontweight='bold')

# Panel 1: Heatmap
ax1 = fig.add_subplot(gs[0, :])
heatmap_data = regional_sig.pivot_table(
    values='lr_means',
    index='interaction',
    columns='region',
    aggfunc='mean'
)
region_order = heatmap_data.sum().sort_values(ascending=False).index
heatmap_data = heatmap_data[region_order]

sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax1,
           cbar_kws={'label': 'Communication Strength'}, linewidths=0.5)
ax1.set_title('Communication Strength by Region', fontsize=13, fontweight='bold')

# Panel 2: Directional count by region
ax2 = fig.add_subplot(gs[1, 0])
direction_counts = regional_sig.groupby(['region', 'direction']).size().unstack(fill_value=0)
direction_counts.plot(kind='barh', stacked=True, ax=ax2, 
                     color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
ax2.set_xlabel('Number of Interactions', fontsize=11, fontweight='bold')
ax2.set_title('Directional Interaction Counts', fontsize=13, fontweight='bold')
ax2.legend(title='Direction', fontsize=9)
ax2.grid(axis='x', alpha=0.3)

# Panel 3: Directional STRENGTH by region - NO LEGEND
ax3 = fig.add_subplot(gs[1, 1])
direction_strength = regional_sig.groupby(['region', 'direction'])['lr_means'].mean().unstack(fill_value=0)
direction_strength.plot(kind='barh', ax=ax3,
                       color=['#FF6B6B', '#4ECDC4'], alpha=0.8, legend=False)
ax3.set_xlabel('Average Communication Strength', fontsize=11, fontweight='bold')
ax3.set_title('Directional Communication Strength', fontsize=13, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Panel 4: Directional BALANCE - arrows showing net direction
ax4 = fig.add_subplot(gs[2, :])

y_positions = np.arange(len(regions_list))

for i, region in enumerate(regions_list):
    reg_data = regional_sig[regional_sig['region'] == region]
    
    # Calculate totals for each direction
    a_to_g = reg_data[reg_data['source'] == 'Astrocyte']
    g_to_a = reg_data[reg_data['source'] == 'GABA_neuron']
    
    a_to_g_strength = a_to_g['lr_means'].sum()
    g_to_a_strength = g_to_a['lr_means'].sum()
    
    # Net balance
    net_balance = a_to_g_strength - g_to_a_strength
    
    y = y_positions[i]
    
    # Draw arrow based on net balance
    if net_balance > 0:  # More Astrocyte → GABA
        arrow_length = min(abs(net_balance) * 0.15, 3)
        ax4.arrow(0.5, y, arrow_length, 0,
                 head_width=0.2, head_length=0.15,
                 fc='#FF6B6B', ec='#FF6B6B',
                 linewidth=3, alpha=0.8)
        ax4.text(0.5 + arrow_length + 0.3, y, f"+{net_balance:.1f}",
                va='center', fontsize=10, fontweight='bold', color='#FF6B6B')
    else:  # More GABA → Astrocyte
        arrow_length = min(abs(net_balance) * 0.15, 3)
        ax4.arrow(0.5, y, -arrow_length, 0,
                 head_width=0.2, head_length=0.15,
                 fc='#4ECDC4', ec='#4ECDC4',
                 linewidth=3, alpha=0.8)
        ax4.text(0.5 - arrow_length - 0.5, y, f"{net_balance:.1f}",
                va='center', fontsize=10, fontweight='bold', color='#4ECDC4')
    
    # Region label
    ax4.text(0.5, y, region, ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5))

# Labels - moved up to avoid overlap
ax4.text(0.1, len(regions_list) + 0.7, 'Astrocyte', fontsize=13, fontweight='bold',
        ha='center', bbox=dict(boxstyle='round', facecolor='#FF6B6B', alpha=0.5))
ax4.text(0.9, len(regions_list) + 0.7, 'GABA', fontsize=13, fontweight='bold',
        ha='center', bbox=dict(boxstyle='round', facecolor='#4ECDC4', alpha=0.5))

ax4.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_xlim(-1, 2)
ax4.set_ylim(-1.2, len(regions_list) + 1.2)  # More space at bottom
ax4.set_title('Net Directional Balance by Region', fontsize=13, fontweight='bold', pad=20)
ax4.text(0.5, -1.0, 'Arrow direction = dominant communication direction\nArrow length & number = strength of imbalance',
        ha='center', fontsize=9, style='italic')
ax4.axis('off')

# Panel 5: Specific interactions by direction
ax5 = fig.add_subplot(gs[3, :])

# Get top interactions for each direction across all regions
top_a_to_g = regional_sig[regional_sig['source'] == 'Astrocyte'].groupby('interaction')['lr_means'].mean().nlargest(3)
top_g_to_a = regional_sig[regional_sig['source'] == 'GABA_neuron'].groupby('interaction')['lr_means'].mean().nlargest(3)

for idx, (interaction, strength) in enumerate(top_a_to_g.items()):
    data = regional_sig[(regional_sig['interaction'] == interaction) & 
                       (regional_sig['source'] == 'Astrocyte')].sort_values('region')
    ax5.plot(data['region'], data['lr_means'], marker='o', linewidth=2, markersize=8,
            color='#FF6B6B', alpha=0.7, label=f"A→G: {interaction[:30]}")

for idx, (interaction, strength) in enumerate(top_g_to_a.items()):
    data = regional_sig[(regional_sig['interaction'] == interaction) & 
                       (regional_sig['source'] == 'GABA_neuron')].sort_values('region')
    ax5.plot(data['region'], data['lr_means'], marker='s', linewidth=2, markersize=8,
            color='#4ECDC4', alpha=0.7, label=f"G→A: {interaction[:30]}")

ax5.set_xlabel('Brain Region', fontsize=11, fontweight='bold')
ax5.set_ylabel('Communication Strength', fontsize=11, fontweight='bold')
ax5.set_title('Top Directional Interactions Across Regions', fontsize=13, fontweight='bold')
ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax5.grid(alpha=0.3)
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('/scratch/easmit31/cell_cell/figures/cellchat_regional_directional_complete.png',
           dpi=300, bbox_inches='tight')
print("\n✓ Saved complete directional analysis")

regional_sig.to_csv('/scratch/easmit31/cell_cell/results/regional_effects_all9_detailed.csv', index=False)
print("✓ Saved detailed table")

