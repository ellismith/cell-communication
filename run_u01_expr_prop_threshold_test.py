#!/usr/bin/env python3
"""
Generate CellChat jobs for ALL individuals and ALL cell types
Testing multiple expr_prop thresholds: 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30
"""
import os
import pandas as pd
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Generate CellChat jobs with multiple expr_prop thresholds')
parser.add_argument('--expr_prop', type=float, required=True, 
                    help='Expression proportion threshold')
args = parser.parse_args()

expr_prop = args.expr_prop
suffix = f"_threshold{expr_prop}".replace('.', 'p')

base_path = "/scratch/nsnyderm/u01/intermediate_files/cell-class_h5ad_update/"
output_dir = f"/scratch/easmit31/cell_cell/results/u01_threshold_test{suffix}"
job_dir = f"/scratch/easmit31/cell_cell/u01_threshold_test_jobs{suffix}"
os.makedirs(job_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

print(f"Configuration:")
print(f"  Expression proportion threshold: {expr_prop}")
print(f"  Output directory: {output_dir}")
print(f"  Job directory: {job_dir}")

# Load expected counts to get list of individuals
expected_counts = pd.read_csv("/scratch/easmit31/cell_cell/u01_cell_counts_per_individual.csv")
all_individuals = sorted(expected_counts['animal_id'].unique())

print(f"\nCreating jobs for {len(all_individuals)} individuals...")

# ALL 11 cell types
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

for individual in all_individuals:
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
adatas = []
for cell_type, filename in cell_types.items():
    adata = sc.read_h5ad(base_path + filename)
    adata = adata[adata.obs['animal_id'] == individual].copy()
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
    print(f"No data for individual {{individual}}")
    exit(0)

adata_combined = sc.concat(adatas, join="inner")
print(f"Combined: {{adata_combined.n_obs:,}} cells, {{adata_combined.n_vars:,}} genes")

# Normalize
adata_combined.X[pd.isna(adata_combined.X)] = 0
sc.pp.normalize_total(adata_combined, target_sum=1e4)
sc.pp.log1p(adata_combined)

# Load interactions with proper complex handling (2,909 interactions)
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
    
    slurm_content = f'''#!/bin/bash
#SBATCH --job-name={individual}{suffix}
#SBATCH --output={job_dir}/{job_name}_%j.out
#SBATCH --error={job_dir}/{job_name}_%j.err
#SBATCH --partition=htc
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

source ~/.bashrc
conda activate cellchat_env

python {script_path}
'''
    
    slurm_path = f"{job_dir}/{job_name}.sh"
    with open(slurm_path, "w") as f:
        f.write(slurm_content)

print(f"\n✓ Created {len(all_individuals)} job scripts with expr_prop={expr_prop}")
print(f"   Output files will be saved as: {{individual}}_results{suffix}.csv")
print(f"\nTo generate jobs for all thresholds:")
print(f"  for thresh in 0.0 0.05 0.10 0.15 0.20 0.25 0.30; do")
print(f"    python run_u01_expr_prop_threshold_test.py --expr_prop $thresh")
print(f"  done")
print(f"\nTo submit all jobs for a specific threshold:")
print(f"  for f in {job_dir}/*.sh; do sbatch $f; done")
