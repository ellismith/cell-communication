#!/usr/bin/env python3
"""
run_pairwise_with_sex.py
-------------------------
Run pairwise cell-cell communication analysis including SEX.
Groups by: cell_type_region_sex
Saves to: /scratch/easmit31/cell_cell/results/pairwise_with_sex/
"""
import itertools
import os
import scanpy as sc

base_path = "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/"
output_dir = "/scratch/easmit31/cell_cell/results/pairwise_with_sex"
job_dir = "pairwise_jobs_with_sex"
os.makedirs(job_dir, exist_ok=True)

cell_types = {
    'Astrocyte': 'adult_grn_astrocytes_selected_converted.h5ad',
    'GABA': 'adult_grn_gaba_selected_converted.h5ad',
    'Glutamatergic': 'adult_grn_glut_selected_converted.h5ad',
    'MSN': 'adult_grn_MSN_selected_converted.h5ad',
    'OPC_Oligo': 'adult_grn_opc-olig_selected_converted.h5ad',
    'Microglia': 'adult_grn_microglia_selected_converted.h5ad'
}

# Check sex distribution
print("Checking sex distribution across cell types...")
for ct, fname in cell_types.items():
    adata = sc.read_h5ad(base_path + fname, backed='r')
    if 'sex' in adata.obs.columns:
        sex_counts = adata.obs['sex'].value_counts()
        print(f"{ct}: {sex_counts.to_dict()}")
    else:
        print(f"⚠️ No sex column in {ct}")
    adata.file.close()

# Generate pairwise jobs
pairs = list(itertools.combinations(cell_types.keys(), 2))
print(f"\nCreating {len(pairs)} jobs with sex stratification...")

for type1, type2 in pairs:
    job_name = f"{type1}_{type2}"
    script_path = f"{job_dir}/{job_name}.py"
    
    script_content = f'''#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import liana as li
import os

print("="*70)
print("Analyzing {type1} ↔ {type2} (with sex)")
print("="*70)

base_path = "{base_path}"
output_dir = "{output_dir}"
os.makedirs(output_dir, exist_ok=True)

# --- Load both cell types
adata1 = sc.read_h5ad(base_path + "{cell_types[type1]}")
adata2 = sc.read_h5ad(base_path + "{cell_types[type2]}")

# --- Tag cell types
adata1.obs["cell_type"] = "{type1}"
adata2.obs["cell_type"] = "{type2}"

# --- Check for sex column
if "sex" not in adata1.obs.columns or "sex" not in adata2.obs.columns:
    print("⚠️ Missing sex column, skipping")
    exit(1)

print(f"  {{adata1.n_obs:,}} {type1} cells, {{adata2.n_obs:,}} {type2} cells")
print(f"  Sex in {type1}: {{adata1.obs['sex'].value_counts().to_dict()}}")
print(f"  Sex in {type2}: {{adata2.obs['sex'].value_counts().to_dict()}}")

# --- Create combined grouping: cell_type + region + sex
adata1.obs["cell_type_region_sex"] = (
    adata1.obs["cell_type"] + "_" + 
    adata1.obs["region"].astype(str) + "_" + 
    adata1.obs["sex"].astype(str)
)
adata2.obs["cell_type_region_sex"] = (
    adata2.obs["cell_type"] + "_" + 
    adata2.obs["region"].astype(str) + "_" + 
    adata2.obs["sex"].astype(str)
)

print(f"  Total groups: {{adata1.obs['cell_type_region_sex'].nunique() + adata2.obs['cell_type_region_sex'].nunique()}}")

# --- Combine
adata_pair = sc.concat([adata1, adata2], join="inner")
print(f"  Combined: {{adata_pair.n_obs:,}} cells, {{adata_pair.n_vars:,}} genes")

# --- Fill NaNs and normalize
adata_pair.X[pd.isna(adata_pair.X)] = 0
sc.pp.normalize_total(adata_pair, target_sum=1e4)
sc.pp.log1p(adata_pair)

# --- Run LIANA
print("\\nRunning CellChat with region + sex preserved...")
li.mt.rank_aggregate(
    adata_pair,
    groupby="cell_type_region_sex",
    resource_name="cellchatdb",
    expr_prop=0.1,
    use_raw=False,
    n_perms=100,
    verbose=True
)

# --- Save results
res = adata_pair.uns["liana_res"].copy()
out_path = os.path.join(output_dir, f"{job_name}_results.csv")
res.to_csv(out_path, index=False)

sig = res[res["magnitude_rank"] < 0.05]
print(f"✓ Saved {{out_path}} ({{len(res):,}} total, {{len(sig):,}} significant)")
'''
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    slurm_content = f'''#!/bin/bash
#SBATCH --job-name={job_name}_sex
#SBATCH --output={job_dir}/{job_name}_%j.out
#SBATCH --error={job_dir}/{job_name}_%j.err
#SBATCH --partition=htc
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

source ~/.bashrc
conda activate cellchat_env
python {script_path}
'''
    
    with open(f"{job_dir}/{job_name}.sh", "w") as f:
        f.write(slurm_content)
    
    os.system(f"sbatch {job_dir}/{job_name}.sh")
    print(f"  Submitted: {job_name}")

print(f"\\n✓ Submitted {len(pairs)} jobs with sex stratification!")
print(f"Results will be in: {output_dir}")
