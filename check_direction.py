import pandas as pd
import glob

files = glob.glob("/scratch/easmit31/cell_cell/results/pairwise/*_results.csv")

print("Checking CX3CL1-CX3CR1 directionality...")
print("Expected: Neuron expresses CX3CL1 (ligand), Microglia expresses CX3CR1 (receptor)\n")

for f in files:
    df = pd.read_csv(f)
    
    df["source_ct"] = df["source"].str.split("_").str[0]
    df["target_ct"] = df["target"].str.split("_").str[0]
    
    # Check correct direction
    correct = df[
        (df["source_ct"].isin(["GABA", "Glutamatergic"])) &
        (df["target_ct"] == "Microglia") &
        (df["ligand_complex"].str.contains("CX3CL1", case=False, na=False)) &
        (df["receptor_complex"].str.contains("CX3CR1", case=False, na=False))
    ]
    
    # Check wrong direction (should not exist!)
    wrong = df[
        (df["source_ct"] == "Microglia") &
        (df["target_ct"].isin(["GABA", "Glutamatergic"])) &
        (df["ligand_complex"].str.contains("CX3CR1", case=False, na=False)) &
        (df["receptor_complex"].str.contains("CX3CL1", case=False, na=False))
    ]
    
    if len(correct) > 0:
        print(f"✓ Correct direction found: {len(correct)} interactions in {f.split('/')[-1]}")
    if len(wrong) > 0:
        print(f"⚠️  WRONG DIRECTION: {len(wrong)} interactions in {f.split('/')[-1]}")

