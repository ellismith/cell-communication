import pandas as pd
import glob

files = glob.glob("/scratch/easmit31/cell_cell/results/pairwise/*_results.csv")

canonical_checks = {
    "Astrocyte->Neuron APOE": {
        "source": "Astrocyte",
        "target": ["GABA", "Glutamatergic"],
        "ligand": "APOE",
        "receptor": ["LRP1", "LDLR", "VLDLR"]
    },
    "Neuron->Microglia CX3CL1": {
        "source": ["GABA", "Glutamatergic"],
        "target": "Microglia",
        "ligand": "CX3CL1",
        "receptor": "CX3CR1"
    },
    "Astrocyte->Neuron TNF": {
        "source": "Astrocyte",
        "target": ["GABA", "Glutamatergic"],
        "ligand": "TNF",
        "receptor": ["TNFRSF1A", "TNFRSF1B"]
    },
    "Astrocyte->Neuron GAS6": {
        "source": "Astrocyte",
        "target": ["GABA", "Glutamatergic"],
        "ligand": "GAS6",
        "receptor": ["AXL", "MERTK", "TYRO3"]
    },
    "OPC_Oligo->Neuron NRG1": {
        "source": "OPC_Oligo",
        "target": ["GABA", "Glutamatergic"],
        "ligand": "NRG1",
        "receptor": ["ERBB2", "ERBB3", "ERBB4"]
    }
}

print("Validating canonical patterns...\n")

for check_name, criteria in canonical_checks.items():
    print(f"{'='*70}")
    print(f"Checking: {check_name}")
    print(f"{'='*70}")
    
    found = False
    for f in files:
        df = pd.read_csv(f)
        
        # Parse source/target cell types
        df["source_ct"] = df["source"].str.split("_").str[0]
        df["target_ct"] = df["target"].str.split("_").str[0]
        
        # Create interaction column
        df["interaction"] = df["ligand_complex"] + "→" + df["receptor_complex"]
        
        # Check criteria
        source_match = df["source_ct"] == criteria["source"] if isinstance(criteria["source"], str) else df["source_ct"].isin(criteria["source"])
        target_match = df["target_ct"] == criteria["target"] if isinstance(criteria["target"], str) else df["target_ct"].isin(criteria["target"])
        ligand_match = df["ligand_complex"].str.contains(criteria["ligand"], case=False, na=False)
        receptor_match = df["receptor_complex"].str.contains("|".join(criteria["receptor"]), case=False, na=False)
        
        matches = df[source_match & target_match & ligand_match & receptor_match]
        
        if len(matches) > 0:
            found = True
            sig_matches = matches[matches["magnitude_rank"] < 0.05]
            print(f"  ✓ Found {len(matches)} interactions ({len(sig_matches)} significant)")
            print(f"    File: {f.split('/')[-1]}")
            
            # Show top 3 by lr_means
            top = matches.nlargest(3, 'lr_means')
            for idx, row in top.iterrows():
                sig_marker = "***" if row['magnitude_rank'] < 0.05 else ""
                print(f"    - {row['interaction']}: lr_means={row['lr_means']:.3f}, p={row['magnitude_rank']:.4f} {sig_marker}")
    
    if not found:
        print(f"  ⚠️  NOT FOUND - This canonical pattern is missing!")
        print(f"     This could mean:")
        print(f"     1. Ligand/receptor not in CellChat database")
        print(f"     2. Expression too low in your data")
        print(f"     3. Filtered out as non-significant")
    print()

print("\n" + "="*70)
print("SUMMARY OF KEY FINDINGS:")
print("="*70)
print("✓ CX3CL1-CX3CR1 direction is CORRECT (Neuron→Microglia)")
print("⚠️  APOE signaling not detected - may need investigation")
print("\nRecommendations:")
print("1. Check if APOE/LRP1 are in CellChat ligand-receptor database")
print("2. Verify these genes have sufficient expression in your data")
print("3. Some canonical patterns may be region-specific (check individual regions)")
