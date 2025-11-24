#!/usr/bin/env python3
"""
Expand CellPhoneDB interactions to include complexes resolved to gene names
"""
import pandas as pd
import itertools

# Load all tables
print("Loading CellPhoneDB tables...")
interaction_df = pd.read_csv("CellPhoneDB_latest/interaction_input_CellPhoneDB.csv")
complex_composition = pd.read_csv("CellPhoneDB-data/complex_composition_table.csv")
multidata = pd.read_csv("CellPhoneDB-data/multidata_table.csv")
protein_table = pd.read_csv("CellPhoneDB-data/protein_table.csv")
gene_input = pd.read_csv("CellPhoneDB-data/data/gene_input.csv")

# Create protein_name to gene_name mapping
print("Creating protein to gene mapping...")
protein_full = protein_table.merge(multidata, left_on='protein_multidata_id', right_on='id_multidata')
protein_genes = protein_full.merge(gene_input, left_on='name', right_on='uniprot', how='left')
protein_to_gene = dict(zip(protein_genes['protein_name'], protein_genes['gene_name']))

# Function to resolve partner to gene name(s)
def resolve_partner(partner_id, protein_name):
    """
    Resolve a partner (protein or complex) to gene name(s)
    Returns list of gene names
    """
    # If it's a simple protein with protein_name, use that
    if pd.notna(protein_name):
        gene = protein_to_gene.get(protein_name)
        return [gene] if gene else []
    
    # Otherwise it's a complex - look it up
    complex_info = multidata[multidata['name'] == partner_id]
    if len(complex_info) == 0:
        return []
    
    complex_multidata_id = complex_info['id_multidata'].iloc[0]
    
    # Get component proteins
    components = complex_composition[complex_composition['complex_multidata_id'] == complex_multidata_id]
    
    genes = []
    for _, comp in components.iterrows():
        protein_multidata_id = comp['protein_multidata_id']
        protein_info = protein_table[protein_table['protein_multidata_id'] == protein_multidata_id]
        
        if len(protein_info) > 0:
            prot_name = protein_info['protein_name'].iloc[0]
            gene = protein_to_gene.get(prot_name)
            if gene:
                genes.append(gene)
    
    return genes

# Expand all interactions
print("\nExpanding interactions...")
expanded_interactions = []

for idx, row in interaction_df.iterrows():
    if idx % 500 == 0:
        print(f"  Processing {idx}/{len(interaction_df)}...")
    
    # Resolve both partners
    genes_a = resolve_partner(row['partner_a'], row['protein_name_a'])
    genes_b = resolve_partner(row['partner_b'], row['protein_name_b'])
    
    # If either side is a complex, create all pairwise combinations
    # If both are simple proteins, just one interaction
    for gene_a in genes_a:
        for gene_b in genes_b:
            expanded_interactions.append({
                'gene_a': gene_a,
                'gene_b': gene_b,
                'partner_a': row['partner_a'],
                'partner_b': row['partner_b'],
                'is_complex_a': pd.isna(row['protein_name_a']),
                'is_complex_b': pd.isna(row['protein_name_b'])
            })

# Convert to dataframe and remove duplicates
expanded_df = pd.DataFrame(expanded_interactions)
print(f"\nTotal expanded interactions: {len(expanded_df)}")

# Keep only unique gene pairs
unique_df = expanded_df[['gene_a', 'gene_b']].drop_duplicates()
print(f"Unique gene pairs: {len(unique_df)}")

# Save
unique_df.to_csv("cellphonedb_interactions_all_with_complexes.csv", index=False)
print(f"\n✓ Saved to cellphonedb_interactions_all_with_complexes.csv")

# Also save the full version with metadata
expanded_df.to_csv("cellphonedb_interactions_expanded_full.csv", index=False)
print(f"✓ Saved detailed version to cellphonedb_interactions_expanded_full.csv")

print(f"\nSummary:")
print(f"  Original interactions: {len(interaction_df)}")
print(f"  After expanding complexes: {len(expanded_df)}")
print(f"  Unique gene pairs: {len(unique_df)}")
print(f"  Previously had (protein-only): 977")
print(f"  Gain from complexes: {len(unique_df) - 977}")
