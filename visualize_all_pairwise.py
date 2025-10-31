#!/usr/bin/env python3
"""
Visualize all pairwise cell-cell communication results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import glob
import os

print("="*70)
print("Loading all pairwise results...")
print("="*70)

# Load all pairwise result files
result_files = glob.glob("/scratch/easmit31/cell_cell/results/pairwise/*_results.csv")
print(f"Found {len(result_files)} result files")

all_results = []
for file in result_files:
    df = pd.read_csv(file)
    all_results.append(df)

combined = pd.concat(all_results, ignore_index=True)
print(f"Total interactions: {len(combined):,}")

# Filter significant
sig = combined[combined['magnitude_rank'] < 0.05].copy()
print(f"Significant interactions: {len(sig):,}")

sig['interaction'] = sig['ligand_complex'] + ' → ' + sig['receptor_complex']

# Summary statistics
print("\n" + "="*70)
print("SUMMARY BY CELL TYPE PAIR")
print("="*70)

pair_summary = sig.groupby(['source', 'target']).agg({
    'interaction': 'count',
    'lr_means': 'mean'
}).rename(columns={'interaction': 'n_interactions', 'lr_means': 'avg_strength'})

print(pair_summary.sort_values('n_interactions', ascending=False))

# Create comprehensive visualization
fig = plt.figure(figsize=(22, 16))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
fig.suptitle('Cell-Cell Communication Across All Cell Types', 
             fontsize=18, fontweight='bold')

# ============================================================================
# Panel 1: Communication network graph
# ============================================================================
ax1 = fig.add_subplot(gs[0, :2])

# Create directed network
G = nx.DiGraph()
cell_types = list(set(sig['source'].unique()) | set(sig['target'].unique()))

# Add nodes
for ct in cell_types:
    G.add_node(ct)

# Add edges
for (source, target), group in sig.groupby(['source', 'target']):
    if source != target:
        n_interactions = len(group)
        avg_strength = group['lr_means'].mean()
        G.add_edge(source, target, weight=n_interactions, strength=avg_strength)

# Draw network
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
node_colors = plt.cm.Set3(np.linspace(0, 1, len(cell_types)))
node_color_map = {ct: node_colors[i] for i, ct in enumerate(cell_types)}

nx.draw_networkx_nodes(G, pos, node_size=3000, 
                       node_color=[node_color_map[node] for node in G.nodes()],
                       alpha=0.9, ax=ax1)

# Draw edges with arrows
for (u, v, d) in G.edges(data=True):
    weight = d['weight']
    ax1.annotate('',
                xy=pos[v], xycoords='data',
                xytext=pos[u], textcoords='data',
                arrowprops=dict(arrowstyle='->', lw=weight/2, 
                              alpha=0.6, color='gray',
                              connectionstyle="arc3,rad=0.1"))

nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', ax=ax1)

ax1.set_title('Communication Network (edge width = # interactions)', 
              fontsize=14, fontweight='bold')
ax1.axis('off')

# ============================================================================
# Panel 2: Top communicating pairs
# ============================================================================
ax2 = fig.add_subplot(gs[0, 2])

top_pairs = sig.groupby(['source', 'target']).size().nlargest(15)
y_pos = np.arange(len(top_pairs))
pair_labels = [f"{src} → {tgt}" for (src, tgt) in top_pairs.index]

colors = plt.cm.viridis(np.linspace(0, 1, len(top_pairs)))
ax2.barh(y_pos, top_pairs.values, color=colors, alpha=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(pair_labels, fontsize=9)
ax2.set_xlabel('Number of Interactions', fontsize=11, fontweight='bold')
ax2.set_title('Top 15 Communicating Pairs', fontsize=13, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# ============================================================================
# Panel 3: Heatmap of interaction counts
# ============================================================================
ax3 = fig.add_subplot(gs[1, :])

interaction_matrix = sig.groupby(['source', 'target']).size().unstack(fill_value=0)
sns.heatmap(interaction_matrix, annot=True, fmt='d', cmap='YlOrRd', ax=ax3,
           cbar_kws={'label': '# Interactions'}, linewidths=0.5)
ax3.set_title('Interaction Count Matrix', fontsize=14, fontweight='bold')
ax3.set_xlabel('Target Cell Type', fontsize=12)
ax3.set_ylabel('Source Cell Type', fontsize=12)

# ============================================================================
# Panel 4: Communication strength heatmap
# ============================================================================
ax4 = fig.add_subplot(gs[2, :2])

strength_matrix = sig.groupby(['source', 'target'])['lr_means'].mean().unstack(fill_value=0)
sns.heatmap(strength_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax4,
           cbar_kws={'label': 'Avg Communication Strength'}, linewidths=0.5)
ax4.set_title('Communication Strength Matrix', fontsize=14, fontweight='bold')
ax4.set_xlabel('Target Cell Type', fontsize=12)
ax4.set_ylabel('Source Cell Type', fontsize=12)

# ============================================================================
# Panel 5: Top ligand-receptor pairs overall
# ============================================================================
ax5 = fig.add_subplot(gs[2, 2])

top_lr = sig.groupby('interaction')['lr_means'].mean().nlargest(10)
y_pos = np.arange(len(top_lr))
lr_labels = [lr[:30] + '...' if len(lr) > 30 else lr for lr in top_lr.index]

ax5.barh(y_pos, top_lr.values, color='coral', alpha=0.8)
ax5.set_yticks(y_pos)
ax5.set_yticklabels(lr_labels, fontsize=8)
ax5.set_xlabel('Avg Strength', fontsize=10, fontweight='bold')
ax5.set_title('Top 10 L-R Pairs\n(across all cell types)', fontsize=12, fontweight='bold')
ax5.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/scratch/easmit31/cell_cell/figures/cellchat_all_pairwise_comprehensive.png',
           dpi=300, bbox_inches='tight')
print("\n✓ Saved comprehensive visualization")

# Save combined results
sig.to_csv('/scratch/easmit31/cell_cell/results/cellchat_all_pairwise_significant.csv', index=False)
combined.to_csv('/scratch/easmit31/cell_cell/results/cellchat_all_pairwise_complete.csv', index=False)
print("✓ Saved combined results")

# Print detailed summary
print("\n" + "="*70)
print("DETAILED SUMMARY")
print("="*70)
print(f"\nCell types analyzed: {len(cell_types)}")
print(f"Total cell type pairs: {len(result_files)}")
print(f"Total significant interactions: {len(sig):,}")
print(f"\nInteractions per source cell type:")
print(sig['source'].value_counts())
print(f"\nInteractions per target cell type:")
print(sig['target'].value_counts())

