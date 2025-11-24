#!/usr/bin/env python3
"""
Compare LIANA's built-in CellChatDB vs your downloaded CellPhoneDB v5
"""
import liana as li
import pandas as pd

print("="*70)
print("1. LIANA's built-in CellChatDB (what you're currently using)")
print("="*70)
cellchatdb = li.rs.select_resource('cellchatdb')
print(f"Interactions: {len(cellchatdb)}")
print(f"Columns: {cellchatdb.columns.tolist()}")
print(f"\nFirst 5 interactions:")
print(cellchatdb.head())

print("\n" + "="*70)
print("2. Your downloaded CellPhoneDB v5 (NOT currently used)")
print("="*70)

# Check which file format is best
cpdb_liana = pd.read_csv('cellphonedb_interactions_liana_format.csv')
print(f"Interactions: {len(cpdb_liana)}")
print(f"Columns: {cpdb_liana.columns.tolist()}")
print(f"\nFirst 5 interactions:")
print(cpdb_liana.head())

print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"CellChatDB (current):     {len(cellchatdb):,} interactions")
print(f"CellPhoneDB v5 (yours):   {len(cpdb_liana):,} interactions")
print(f"Difference:               {len(cpdb_liana) - len(cellchatdb):,} more in CellPhoneDB v5")

# Check for neurotransmitter-specific interactions
for db_name, db in [('CellChatDB', cellchatdb), ('CellPhoneDB v5', cpdb_liana)]:
    print(f"\n{db_name} - Neurotransmitter interactions:")
    for signal_type, pattern in [
        ('Dopaminergic', 'DRD|DDC|TH|SLC6A3'),
        ('Glutamatergic', 'GRIA|GRIN|GRM'),
        ('GABAergic', 'GABR|GAD'),
        ('Synaptic', 'NLGN|NRXN')
    ]:
        matches = db[
            db['ligand'].str.contains(pattern, case=False, regex=True, na=False) |
            db['receptor'].str.contains(pattern, case=False, regex=True, na=False)
        ]
        print(f"  {signal_type}: {len(matches)}")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("Your downloaded CellPhoneDB v5 has MORE interactions.")
print("To use it, you need to modify run_u01_per_individual.py to load")
print("the custom database instead of using resource_name='cellchatdb'")
print("\nWant me to show you how to switch to CellPhoneDB v5?")
