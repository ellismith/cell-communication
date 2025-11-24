#!/usr/bin/env python3
"""
Create summary tables showing top 5 interactions per individual per region
"""
import pandas as pd
import os
from pathlib import Path

results_dir = "results/u01_per_individual"
output_dir = "results/u01_individual_summaries"
os.makedirs(output_dir, exist_ok=True)

result_files = [f for f in os.listdir(results_dir) 
                if f.endswith('_results.csv') and 'ALL_INTERACTIONS' not in f]

print(f"Found {len(result_files)} individuals")

# Load metadata
metadata = pd.read_csv("results/u01_per_individual_summary.csv")
print(f"Loaded metadata for {len(metadata)} individuals")

all_within_summaries = []
all_cross_summaries = []

for result_file in result_files:
    animal_id = result_file.split('_')[0]
    print(f"\nProcessing {animal_id}...")
    
    df = pd.read_csv(f"{results_dir}/{result_file}")
    
    # Parse regions and cell types
    df['source_celltype'] = df['source'].str.rsplit('_', n=1).str[0]
    df['target_celltype'] = df['target'].str.rsplit('_', n=1).str[0]
    df['source_region'] = df['source'].str.rsplit('_', n=1).str[1]
    df['target_region'] = df['target'].str.rsplit('_', n=1).str[1]
    
    df['celltype_pair'] = df['source_celltype'] + ' -> ' + df['target_celltype']
    df['region_pair'] = df['source_region'] + ' -> ' + df['target_region']
    df['lr_pair'] = df['ligand_complex'] + ' -> ' + df['receptor_complex']
    
    # Get metadata for this individual
    ind_meta = metadata[metadata['animal_id'] == animal_id]
    if len(ind_meta) > 0:
        age = ind_meta['age'].values[0]
        sex = ind_meta['sex'].values[0]
    else:
        age = df['age'].values[0] if 'age' in df.columns else 'Unknown'
        sex = df['sex'].values[0] if 'sex' in df.columns else 'Unknown'
    
    # WITHIN-REGION: Top 5 per region per celltype_pair
    within = df[df['source_region'] == df['target_region']].copy()
    
    for region in within['source_region'].unique():
        region_data = within[within['source_region'] == region]
        
        for celltype_pair in region_data['celltype_pair'].unique():
            celltype_data = region_data[region_data['celltype_pair'] == celltype_pair]
            
            # Get top 5 by lr_means
            top5 = celltype_data.nlargest(5, 'lr_means')
            
            for rank, (idx, row) in enumerate(top5.iterrows(), 1):
                all_within_summaries.append({
                    'animal_id': animal_id,
                    'age': age,
                    'sex': sex,
                    'region': region,
                    'celltype_pair': celltype_pair,
                    'rank': rank,
                    'lr_pair': row['lr_pair'],
                    'lr_means': row['lr_means'],
                    'qval': row['qval']
                })
    
    # CROSS-REGION: Top 5 per region_pair per celltype_pair
    cross = df[df['source_region'] != df['target_region']].copy()
    
    for region_pair in cross['region_pair'].unique():
        region_data = cross[cross['region_pair'] == region_pair]
        
        for celltype_pair in region_data['celltype_pair'].unique():
            celltype_data = region_data[region_data['celltype_pair'] == celltype_pair]
            
            # Get top 5 by lr_means
            top5 = celltype_data.nlargest(5, 'lr_means')
            
            for rank, (idx, row) in enumerate(top5.iterrows(), 1):
                all_cross_summaries.append({
                    'animal_id': animal_id,
                    'age': age,
                    'sex': sex,
                    'region_pair': region_pair,
                    'celltype_pair': celltype_pair,
                    'rank': rank,
                    'lr_pair': row['lr_pair'],
                    'lr_means': row['lr_means'],
                    'qval': row['qval']
                })

# Convert to DataFrames
within_summary = pd.DataFrame(all_within_summaries)
cross_summary = pd.DataFrame(all_cross_summaries)

# Save
within_summary.to_csv(f"{output_dir}/top5_within_region_per_individual.csv", index=False)
cross_summary.to_csv(f"{output_dir}/top5_cross_region_per_individual.csv", index=False)

print(f"\n✓ Saved summaries:")
print(f"  Within-region: {len(within_summary)} rows")
print(f"  Cross-region: {len(cross_summary)} rows")

# Create pivot tables for easier viewing
print("\nCreating pivot tables...")

# Within-region: Show which regions each individual has
within_pivot = within_summary.groupby(['animal_id', 'region']).size().reset_index(name='n_interactions')
within_pivot = within_pivot.pivot(index='animal_id', columns='region', values='n_interactions').fillna(0)
within_pivot.to_csv(f"{output_dir}/individual_by_region_matrix.csv")
print(f"  Individual x Region matrix: {output_dir}/individual_by_region_matrix.csv")

# Cross-region: Show which region pairs each individual has
cross_pivot = cross_summary.groupby(['animal_id', 'region_pair']).size().reset_index(name='n_interactions')
cross_top_pairs = cross_pivot.groupby('region_pair').size().nlargest(50).index
cross_pivot_filtered = cross_pivot[cross_pivot['region_pair'].isin(cross_top_pairs)]
cross_pivot_wide = cross_pivot_filtered.pivot(index='animal_id', columns='region_pair', values='n_interactions').fillna(0)
cross_pivot_wide.to_csv(f"{output_dir}/individual_by_region_pair_matrix_top50.csv")
print(f"  Individual x Region_pair matrix (top 50): {output_dir}/individual_by_region_pair_matrix_top50.csv")

print("\n✓ Done!")
