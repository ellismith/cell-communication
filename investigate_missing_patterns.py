import pandas as pd
import glob
import scanpy as sc

print("INVESTIGATION 1: What ligands/receptors ARE in CellChat results?")
print("="*70)

files = glob.glob("/scratch/easmit31/cell_cell/results/pairwise/*_results.csv")

# Get all unique ligands and receptors
all_ligands = set()
all_receptors = set()

for f in files[:3]:  # Check first 3 files
    df = pd.read_csv(f)
    all_ligands.update(df['ligand_complex'].unique())
    all_receptors.update(df['receptor_complex'].unique())

print(f"Total unique ligands found: {len(all_ligands)}")
print(f"Total unique receptors found: {len(all_receptors)}")

# Check for our missing genes
missing_genes = ['APOE', 'LRP1', 'LDLR', 'TNF', 'TNFRSF1A', 'GAS6', 'AXL', 'NRG1', 'ERBB3']
print(f"\nSearching for missing canonical genes in CellChat results:")
for gene in missing_genes:
    in_ligands = any(gene in lig for lig in all_ligands)
    in_receptors = any(gene in rec for rec in all_receptors)
    status = []
    if in_ligands:
        status.append("LIGAND")
    if in_receptors:
        status.append("RECEPTOR")
    if status:
        print(f"  ✓ {gene}: Found as {', '.join(status)}")
    else:
        print(f"  ✗ {gene}: NOT in CellChat results")

print("\n" + "="*70)
print("INVESTIGATION 2: Are these genes expressed in raw data?")
print("="*70)

# Check expression in one h5ad file
adata = sc.read_h5ad("/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/adult_grn_astrocytes_selected_converted.h5ad", backed='r')

print(f"\nChecking Astrocyte data (n={adata.n_obs} cells):")
for gene in ['APOE', 'TNF', 'GAS6']:
    if gene in adata.var_names:
        # Get expression levels
        expr = adata[:, gene].X
        if hasattr(expr, 'toarray'):
            expr = expr.toarray().flatten()
        mean_expr = expr.mean()
        pct_expr = (expr > 0).mean() * 100
        print(f"  {gene}: mean={mean_expr:.3f}, expressed in {pct_expr:.1f}% of cells")
    else:
        print(f"  {gene}: NOT FOUND in gene list")

adata.file.close()

print("\n" + "="*70)
print("INVESTIGATION 3: What ARE the top Astrocyte→Neuron interactions?")
print("="*70)

for f in files:
    if 'Astrocyte_GABA' in f or 'Astrocyte_Glutamatergic' in f:
        df = pd.read_csv(f)
        sig = df[df['magnitude_rank'] < 0.05]
        
        print(f"\nFile: {f.split('/')[-1]}")
        print(f"Total interactions: {len(df)}, Significant: {len(sig)}")
        
        if len(sig) > 0:
            print("Top 5 significant interactions:")
            top5 = sig.nlargest(5, 'lr_means')
            for idx, row in top5.iterrows():
                print(f"  {row['ligand_complex']}→{row['receptor_complex']}: lr_means={row['lr_means']:.3f}, p={row['magnitude_rank']:.4f}")
        else:
            print("  No significant interactions!")
            print("Top 5 by lr_means (non-significant):")
            top5 = df.nlargest(5, 'lr_means')
            for idx, row in top5.iterrows():
                print(f"  {row['ligand_complex']}→{row['receptor_complex']}: lr_means={row['lr_means']:.3f}, p={row['magnitude_rank']:.4f}")

print("\n" + "="*70)
print("INVESTIGATION 4: CX3CL1 found but not significant - why?")
print("="*70)

for f in files:
    if 'GABA_Microglia' in f:
        df = pd.read_csv(f)
        cx3cl1 = df[df['ligand_complex'].str.contains('CX3CL1', case=False, na=False)]
        
        if len(cx3cl1) > 0:
            print(f"\nCX3CL1 in {f.split('/')[-1]}:")
            print(f"  Found {len(cx3cl1)} interactions across {cx3cl1['source'].nunique()} source-target pairs")
            print(f"  lr_means range: {cx3cl1['lr_means'].min():.3f} - {cx3cl1['lr_means'].max():.3f}")
            print(f"  p-value range: {cx3cl1['magnitude_rank'].min():.3f} - {cx3cl1['magnitude_rank'].max():.3f}")
            print(f"  Best p-value: {cx3cl1['magnitude_rank'].min():.3f} (threshold: 0.05)")
            print(f"\n  Interpretation: CX3CL1 expression is PRESENT but below significance threshold")
            print(f"  This could mean:")
            print(f"    - Expression is modest but real")
            print(f"    - High variability across regions")
            print(f"    - Conservative permutation test")
        break

