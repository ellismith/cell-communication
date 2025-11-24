#!/usr/bin/env python3
"""
Regenerate jobs for individual 2C0 only with 256GB memory
"""
import os

base_path = "/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/"
base_output_dir = "/scratch/easmit31/cell_cell/results/expr_prop_test"
base_job_dir = "/scratch/easmit31/cell_cell/expr_prop_test_jobs"

individual = "2C0"

# Only these two cell types
cell_types = {
    'GABA': 'Res1_GABAergic-neurons_subset.h5ad',
    'Glutamatergic': 'Res1_glutamatergic-neurons_update.h5ad'
}

# Test these thresholds
expr_prop_thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

print(f"Regenerating jobs for individual {individual} with 256GB memory...")

for expr_prop in expr_prop_thresholds:
    suffix = f"_filtering{expr_prop}".replace('.', 'p')
    output_dir = f"{base_output_dir}/results{suffix}"
    job_dir = f"{base_job_dir}/jobs{suffix}"
    
    job_name = f"ind_{individual}{suffix}"
    script_path = f"{job_dir}/{job_name}.py"
    
    # Build files dict
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

individual = "{individual}"
base_path = "{base_path}"
output_dir = "{output_dir}"
expr_prop = {expr_prop}
os.makedirs(output_dir, exist_ok=True)

cell_types = {files_dict_str}

print(f"Loading data for individual {{individual}}...")
print(f"Expression proportion threshold: {{expr_prop}}")
print(f"Subsetting to HIP and ACC regions only")

adatas = []
for cell_type, filename in cell_types.items():
    adata = sc.read_h5ad(base_path + filename)
    
    # Filter to individual AND HIP/ACC regions only
    adata = adata[
        (adata.obs['animal_id'] == individual) & 
        (adata.obs['region'].isin(['HIP', 'ACC']))
    ].copy()
    
    if adata.n_obs == 0:
        continue
    
    has_name = ~adata.var['external_gene_name'].isna()
    adata = adata[:, has_name].copy()
    adata.var_names = adata.var['external_gene_name'].astype(str).values
    adata.var_names_make_unique()
    
    adata.obs["cell_type"] = cell_type
    adata.obs["cell_type_region"] = adata.obs["cell_type"] + "_" + adata.obs["region"].astype(str)
    adatas.append(adata)

if len(adatas) == 0:
    print(f"No data for individual {{individual}} in HIP/ACC")
    exit(0)

adata_combined = sc.concat(adatas, join="inner")
print(f"Combined: {{adata_combined.n_obs:,}} cells, {{adata_combined.n_vars:,}} genes")

# Normalize
adata_combined.X[pd.isna(adata_combined.X)] = 0
sc.pp.normalize_total(adata_combined, target_sum=1e4)
sc.pp.log1p(adata_combined)

# Load interactions
print("Loading CellPhoneDB interactions...")
interactions_df = pd.read_csv("/scratch/easmit31/cell_cell/cellphonedb_interactions_liana_format.csv")
interactions_list = list(zip(interactions_df['ligand'], interactions_df['receptor']))
print(f"Loaded {{len(interactions_list)}} interactions")

# Run LIANA
print(f"Running LIANA for individual {{individual}} with expr_prop={{expr_prop}}...")
li.mt.rank_aggregate(
    adata_combined,
    groupby="cell_type_region",
    interactions=interactions_list,
    expr_prop=expr_prop,
    use_raw=False,
    n_perms=100,
    verbose=True
)

# Save results with FDR
print("Saving results...")
res = adata_combined.uns["liana_res"].copy()
_, res['qval'], _, _ = multipletests(res['magnitude_rank'], method='fdr_bh')

res['animal_id'] = individual
res['age'] = adata_combined.obs['age'].iloc[0]
res['sex'] = adata_combined.obs['sex'].iloc[0]
res['expr_prop_filter'] = expr_prop

out_path = os.path.join(output_dir, f"{{individual}}_results{suffix}.csv")
res.to_csv(out_path, index=False)

sig = res[res["qval"] < 0.05]
print(f"\\n✓ Saved {{out_path}}")
print(f"  Individual: {{individual}}, Age: {{res['age'].iloc[0]}}, Sex: {{res['sex'].iloc[0]}}")
print(f"  Expression filter: {{expr_prop}}")
print(f"  Total interactions: {{len(res):,}}")
print(f"  Significant (q<0.05): {{len(sig):,}}")
'''
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # HIGH MEMORY VERSION
    slurm_content = f'''#!/bin/bash
#SBATCH --job-name={individual}{suffix}_highmem
#SBATCH --output={job_dir}/{job_name}_highmem_%j.out
#SBATCH --error={job_dir}/{job_name}_highmem_%j.err
#SBATCH --partition=htc
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

source ~/.bashrc
conda activate cellchat_env

python {script_path}
'''
    
    slurm_path = f"{job_dir}/{job_name}_highmem.sh"
    with open(slurm_path, "w") as f:
        f.write(slurm_content)
    
    print(f"✓ Created job for expr_prop={expr_prop}")

print(f"\n{'='*60}")
print(f"Created 7 high-memory jobs for individual {individual}")
print(f"Memory: 256GB, Time: 3 hours")
print(f"\nTo submit all:")
print(f"  for f in {base_job_dir}/jobs_filtering*/*_highmem.sh; do sbatch $f; done")
