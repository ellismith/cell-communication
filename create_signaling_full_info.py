#!/usr/bin/env python3
"""
Create communication matrices with BOTH cell type AND region information
"""
import pandas as pd
import numpy as np
import os

# Define signaling types
signaling_keywords = {
    'Glutamatergic': ['GRIA', 'GRIN', 'GRM', 'GLUT', 'SLC17A'],
    'GABAergic': ['GABR', 'GABA', 'SLC6A1', 'GAD'],
    'Serotonergic': ['HTR', '5HT', 'SLC6A4'],
    'Cholinergic': ['CHRN', 'CHRM', 'ACHE', 'CHAT'],
    'Noradrenergic': ['ADR', 'SLC6A2'],
    'Dopaminergic': ['DRD', 'SLC6A3', 'TH', 'DDC'],
    'Synaptic': ['NLGN', 'NRXN', 'LRRTM', 'CLSTN'],
    'Growth_factors': ['NRG', 'BDNF', 'NGF', 'GDNF', 'FGF', 'EGF'],
    'Inflammatory': ['IL1', 'IL6', 'TNF', 'CCL', 'CXCL'],
    'Adhesion': ['CDH', 'CADM', 'ITGA', 'ITGB']
}

# Load all data
results_dir = "results/u01_per_individual"
result_files = [f for f in os.listdir(results_dir) 
                if f.endswith('_results.csv') and 'ALL_INTERACTIONS' not in f]

print(f"Loading {len(result_files)} individuals...")

all_data = []
for idx, result_file in enumerate(result_files):
    if idx % 10 == 0:
        print(f"  Loading {idx}/{len(result_files)}...")
    df = pd.read_csv(f"{results_dir}/{result_file}")
    df = df[['source', 'target', 'ligand_complex', 'receptor_complex', 
             'lr_means', 'animal_id', 'age']]
    all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)
print(f"Loaded {len(combined):,} total interactions")

# Parse BOTH cell types AND regions
combined['source_celltype'] = combined['source'].str.rsplit('_', n=1).str[0]
combined['target_celltype'] = combined['target'].str.rsplit('_', n=1).str[0]
combined['source_region'] = combined['source'].str.rsplit('_', n=1).str[1]
combined['target_region'] = combined['target'].str.rsplit('_', n=1).str[1]

# Age groups with numeric prefix
def assign_age_group(age):
    if age < 2:
        return '1_Infant'
    elif age < 8:
        return '2_Young'
    elif age < 15:
        return '3_Middle'
    else:
        return '4_Old'

combined['age_group'] = combined['age'].apply(assign_age_group)

print("\nAge group distribution:")
print(combined.groupby('age_group')['animal_id'].nunique())

# Create output
output_base = "results/u01_signaling_full_info"
os.makedirs(output_base, exist_ok=True)

# Process each signaling type
for signal_type, keywords in signaling_keywords.items():
    print(f"\n{'='*70}")
    print(f"Processing {signal_type}")
    print(f"{'='*70}")
    
    # Filter to this signaling type
    pattern = '|'.join(keywords)
    signal_data = combined[
        combined['ligand_complex'].str.contains(pattern, case=False, regex=True) |
        combined['receptor_complex'].str.contains(pattern, case=False, regex=True)
    ].copy()
    
    print(f"Found {len(signal_data):,} interactions for {signal_type}")
    
    if len(signal_data) == 0:
        print(f"  Skipping {signal_type} - no data")
        continue
    
    # Create directory
    signal_dir = f"{output_base}/{signal_type}"
    os.makedirs(signal_dir, exist_ok=True)
    
    # For each age group
    for age_group in ['1_Infant', '2_Young', '3_Middle', '4_Old']:
        age_data = signal_data[signal_data['age_group'] == age_group]
        
        if len(age_data) == 0:
            continue
        
        # Within-region
        within_data = age_data[age_data['source_region'] == age_data['target_region']]
        if len(within_data) > 0:
            # Group by source_celltype, source_region, target_celltype, target_region
            comm_matrix = within_data.groupby([
                'source_celltype', 'source_region', 
                'target_celltype', 'target_region'
            ])['lr_means'].mean().reset_index()
            
            output_file = f"{signal_dir}/{age_group}_within.csv"
            comm_matrix.to_csv(output_file, index=False)
            print(f"  {age_group} within: {len(comm_matrix)} connections")
        
        # Between-region
        between_data = age_data[age_data['source_region'] != age_data['target_region']]
        if len(between_data) > 0:
            comm_matrix = between_data.groupby([
                'source_celltype', 'source_region',
                'target_celltype', 'target_region'
            ])['lr_means'].mean().reset_index()
            
            output_file = f"{signal_dir}/{age_group}_between.csv"
            comm_matrix.to_csv(output_file, index=False)
            print(f"  {age_group} between: {len(comm_matrix)} connections")

print(f"\n✓ Saved full matrices to {output_base}/")
print("Now you can visualize with cell types, regions, or both!")
