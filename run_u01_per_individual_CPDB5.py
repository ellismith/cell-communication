#!/usr/bin/env python3
"""
Generate CCC jobs for EACH INDIVIDUAL using CellPhoneDB v5.0
Results saved with _CPDB5 suffix to distinguish from old results
"""
import itertools
import os
import scanpy as sc
import pandas as pd

base_path = "/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/"
output_dir = "/scratch/easmit31/cell_cell/results/u01_per_individual_CPDB5"
job_dir = "u01_per_individual_jobs_CPDB5"
os.makedirs(job_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Load expected cell counts
expected_counts_path = "/scratch/easmit31/cell_cell/u01_cell_counts_per_individual.csv"
if not os.path.exists(expected_counts_path):
    print(f"ERROR: Expected counts file not found at {expected_counts_path}")
    exit(1)

expected_counts = pd.read_csv(expected_counts_path)
print(f"Loaded expected cell counts for {expected_counts['animal_id'].nunique()} individuals")

cell_types = {
    'Astrocyte': 'Res1_astrocytes_update.h5ad',
    'Basket': 'Res1_basket-cells_update.h5ad',
    'Cerebellar': 'Res1_cerebellar-neurons_subset.h5ad',
    'Ependymal': 'Res1_ependymal-cells_new.h5ad',
    'GABA': 'Res1_GABAergic-neurons_subset.h5ad',
    'Glutamatergic': 'Res1_glutamatergic-neurons_update.h5ad',
    'MSN': 'Res1_medium-spiny-neurons_subset.h5ad',
    'Microglia': 'Res1_microglia_new.h5ad',
    'Midbrain': 'Res1_midbrain-neurons_update.h5ad',
    'OPC_Oligo': 'Res1_opc-olig_subset.h5ad',
    'Vascular': 'Res1_vascular-cells_subset.h5ad'
}

# Get list of all individuals
all_individuals = sorted(expected_counts['animal_id'].unique())
print(f"Creating jobs for {len(all_individuals)} individuals")
print(f"Using CellPhoneDB v5.0 database")

# Create one job per individual
for individual in all_individuals:
    job_name = f"individual_{individual}_CPDB5"
    script_path = f"{job_dir}/{job_name}.py"
    
    # Build files dict string
    files_dict_str = "{\n"
    for ct, filename in cell_types.items():
        files_dict_str += f"    '{ct}': '{filename}',\n"
    files_dict_str += "}"
    
    script_content = f'''#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import numpy as np
import liana as li
from statsmodels.stats.multitest import multipletests
import os

print("="*70)
print("Analyzing Individual: {individual} with CellPhoneDB v5.0")
print("="*70)

base_path = "{base_path}"
output_dir = "{output_dir}"
os.makedirs(output_dir, exist_ok=True)

# Load CellPhoneDB v5.0
print("\\nLoading CellPhoneDB v5.0 database...")
cpdb5_path = "/scratch/easmit31/cell_cell/cellphonedb_interactions_liana_format.csv"
cpdb5_db = pd.read_csv(cpdb5_path)
print(f"  Loaded {{len(cpdb5_db):,}} interactions from CellPhoneDB v5.0")

# Load expected counts for validation
expected_counts = pd.read_csv("{expected_counts_path}")
expected = expected_counts[expected_counts['animal_id'] == "{individual}"].copy()
print(f"\\nExpected cell counts for individual {individual}:")
print(expected[['cell_type', 'region', 'n_cells']].to_string(index=False))

cell_types = {files_dict_str}

# Load all cell types for this individual WITH VALIDATION
adatas = []
validation_passed = True

for cell_type, filename in cell_types.items():
    print(f"\\n{{'='*70}}")
    print(f"Loading {{cell_type}}...")
    print(f"{{'='*70}}")
    
    adata = sc.read_h5ad(base_path + filename)
    print(f"  Total cells in file: {{adata.n_obs:,}}")
    
    # Filter to this individual only
    adata = adata[adata.obs['animal_id'] == "{individual}"].copy()
    
    print(f"  Cells after filtering to {individual}: {{adata.n_obs:,}}")
    
    if adata.n_obs == 0:
        print(f"  → No cells for {individual}, skipping {{cell_type}}")
        continue
    
    # VALIDATION: Check counts per region match expected
    actual_counts = adata.obs.groupby('region', observed=True).size().to_dict()
    expected_for_type = expected[expected['cell_type'] == cell_type]
    
    print(f"  \\n  VALIDATION CHECK:")
    all_regions = set(actual_counts.keys()) | set(expected_for_type['region'].values)
    
    for region in sorted(all_regions):
        actual = actual_counts.get(region, 0)
        exp_row = expected_for_type[expected_for_type['region'] == region]
        expected_val = exp_row['n_cells'].iloc[0] if len(exp_row) > 0 else 0
        
        match = "✓" if actual == expected_val else "✗ MISMATCH"
        print(f"    {{region}}: actual={{actual}}, expected={{expected_val}} {{match}}")
        
        if actual != expected_val:
            validation_passed = False
    
    # Convert gene names - FIX THE CATEGORICAL ISSUE
    has_name = ~adata.var['external_gene_name'].isna()
    adata = adata[:, has_name].copy()
    
    # Convert to string first to avoid categorical issues
    adata.var_names = adata.var['external_gene_name'].astype(str).values
    adata.var_names_make_unique()
    
    # Tag with cell type
    adata.obs["cell_type"] = cell_type
    adata.obs["cell_type_region"] = adata.obs["cell_type"] + "_" + adata.obs["region"].astype(str)
    
    adatas.append(adata)

# Check if validation passed
if not validation_passed:
    print(f"\\n{{'='*70}}")
    print("ERROR: VALIDATION FAILED!")
    print("Cell counts do not match expected values.")
    print("Please review the mismatches above.")
    print(f"{{'='*70}}")
    exit(1)

print(f"\\n{{'='*70}}")
print("✓ VALIDATION PASSED - All cell counts match expected values")
print(f"{{'='*70}}")

if len(adatas) == 0:
    print(f"\\nNo data for individual {individual}")
    exit(0)

print(f"\\nCombining {{len(adatas)}} cell types...")
adata_combined = sc.concat(adatas, join="inner")
print(f"  Total: {{adata_combined.n_obs:,}} cells, {{adata_combined.n_vars:,}} genes")
print(f"  Cell types present: {{sorted(adata_combined.obs['cell_type'].unique())}}")
print(f"  Regions present: {{sorted(adata_combined.obs['region'].unique())}}")

# Normalize
adata_combined.X[pd.isna(adata_combined.X)] = 0
sc.pp.normalize_total(adata_combined, target_sum=1e4)
sc.pp.log1p(adata_combined)

# Run LIANA with CellPhoneDB v5.0 database
print("\\nRunning LIANA with CellPhoneDB v5.0...")
li.mt.rank_aggregate(
    adata_combined,
    groupby="cell_type_region",
    resource=cpdb5_db,
    expr_prop=0.1,
    use_raw=False,
    n_perms=100,
    verbose=True
)

# Save results with FDR correction
print("\\nApplying FDR correction...")
res = adata_combined.uns["liana_res"].copy()
_, res['qval'], _, _ = multipletests(res['magnitude_rank'], method='fdr_bh')

# Add individual metadata
res['animal_id'] = "{individual}"
age = adata_combined.obs['age'].iloc[0]
sex = adata_combined.obs['sex'].iloc[0]
res['age'] = age
res['sex'] = sex
res['database'] = 'CellPhoneDB_v5.0'

# Save louvain info if exists
if 'louvain' in adata_combined.obs.columns:
    print("\\nSaving louvain cluster information...")
    louvain_summary = adata_combined.obs.groupby(['cell_type', 'region', 'louvain'], observed=True).size().reset_index(name='n_cells_in_cluster')
    louvain_summary['animal_id'] = "{individual}"
    louvain_path = os.path.join(output_dir, f"{individual}_louvain_summary_CPDB5.csv")
    louvain_summary.to_csv(louvain_path, index=False)
    print(f"  Saved louvain summary to {{louvain_path}}")

out_path = os.path.join(output_dir, f"{individual}_results_CPDB5.csv")
res.to_csv(out_path, index=False)

sig = res[res["qval"] < 0.05]
print(f"\\n✓ Saved {{out_path}}")
print(f"  Individual: {individual}, Age: {{age}}, Sex: {{sex}}")
print(f"  Database: CellPhoneDB v5.0")
print(f"  Total interactions: {{len(res):,}}")
print(f"  Significant (FDR q<0.05): {{len(sig):,}}")
'''
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    slurm_content = f'''#!/bin/bash
#SBATCH --job-name={individual}_CPDB5
#SBATCH --output={job_dir}/{job_name}_%j.out
#SBATCH --error={job_dir}/{job_name}_%j.err
#SBATCH --partition=htc
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

source ~/.bashrc
conda activate cellchat_env

python {script_path}
'''
    
    slurm_path = f"{job_dir}/{job_name}.sh"
    with open(slurm_path, "w") as f:
        f.write(slurm_content)

print(f"\n✓ Created {len(all_individuals)} job scripts using CellPhoneDB v5.0")
print(f"   Results will be saved to: {output_dir}/")
print(f"   File names: {{animal_id}}_results_CPDB5.csv")
print(f"\nTo submit all jobs:")
print(f"  cd /scratch/easmit31/cell_cell")
print(f"  for f in {job_dir}/*.sh; do sbatch $f; done")
