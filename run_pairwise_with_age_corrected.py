#!/usr/bin/env python3
"""
Run pairwise CellChat WITH age stratification
- Groups by cell_type + region + age_bin
- ADDED: FDR correction for multiple testing across age groups
"""
import itertools
import os
import scanpy as sc
import numpy as np

base_path = "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/"
output_dir = "/scratch/easmit31/cell_cell/results/pairwise_with_age"
job_dir = "pairwise_jobs_with_age"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(job_dir, exist_ok=True)

cell_types = {
    'Astrocyte': 'adult_grn_astrocytes_selected_converted.h5ad',
    'GABA': 'adult_grn_gaba_selected_converted.h5ad',
    'Glutamatergic': 'adult_grn_glut_selected_converted.h5ad',
    'MSN': 'adult_grn_MSN_selected_converted.h5ad',
    'OPC_Oligo': 'adult_grn_opc-olig_selected_converted.h5ad',
    'Microglia': 'adult_grn_microglia_selected_converted.h5ad'
}

# Check age distribution
print("Checking age distribution...")
all_ages = []
for name, fname in cell_types.items():
    adata = sc.read_h5ad(base_path + fname)
    all_ages.extend(adata.obs['age'].values)
    print(f"  {name}: {len(adata.obs)} cells, age range {adata.obs['age'].min():.1f}-{adata.obs['age'].max():.1f}")

all_ages = np.array(all_ages)
young_max = np.percentile(all_ages, 33)
old_min = np.percentile(all_ages, 67)

print(f"\nAge bins: young (<{young_max:.1f}), medium ({young_max:.1f}-{old_min:.1f}), old (>={old_min:.1f})")

pairs = list(itertools.combinations(cell_types.keys(), 2))
print(f"\nCreating {len(pairs)} jobs with age stratification...")

for type1, type2 in pairs:
    job_name = f"{type1}_{type2}"
    script_path = f"{job_dir}/{job_name}.py"
    
    script_content = f'''#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import numpy as np
import liana as li
from statsmodels.stats.multitest import multipletests
import os

print("="*70)
print("Analyzing {type1} ↔ {type2} (with age)")
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

# --- Check for age column
if "age" not in adata1.obs.columns or "age" not in adata2.obs.columns:
    print("⚠️ Missing age column, skipping")
    exit(1)

# --- Create age bins
young_max = {young_max}
old_min = {old_min}

def assign_age_bin(age):
    if age < young_max:
        return "young"
    elif age < old_min:
        return "medium"
    else:
        return "old"

adata1.obs["age_bin"] = adata1.obs["age"].apply(assign_age_bin)
adata2.obs["age_bin"] = adata2.obs["age"].apply(assign_age_bin)

print(f"  {{adata1.n_obs:,}} {type1} cells, {{adata2.n_obs:,}} {type2} cells")
print(f"  Age in {type1}: {{adata1.obs['age_bin'].value_counts().to_dict()}}")
print(f"  Age in {type2}: {{adata2.obs['age_bin'].value_counts().to_dict()}}")

# --- Create combined grouping: cell_type + region + age
adata1.obs["cell_type_region_age"] = (
    adata1.obs["cell_type"] + "_" + 
    adata1.obs["region"].astype(str) + "_" + 
    adata1.obs["age_bin"].astype(str)
)
adata2.obs["cell_type_region_age"] = (
    adata2.obs["cell_type"] + "_" + 
    adata2.obs["region"].astype(str) + "_" + 
    adata2.obs["age_bin"].astype(str)
)

print(f"  Total groups: {{adata1.obs['cell_type_region_age'].nunique() + adata2.obs['cell_type_region_age'].nunique()}}")

# --- Combine
adata_pair = sc.concat([adata1, adata2], join="inner")
print(f"  Combined: {{adata_pair.n_obs:,}} cells, {{adata_pair.n_vars:,}} genes")

# --- Fill NaNs and normalize
adata_pair.X[pd.isna(adata_pair.X)] = 0
sc.pp.normalize_total(adata_pair, target_sum=1e4)
sc.pp.log1p(adata_pair)

# --- Run LIANA
print("\\nRunning CellChat with region + age preserved...")
li.mt.rank_aggregate(
    adata_pair,
    groupby="cell_type_region_age",
    resource_name="cellchatdb",
    expr_prop=0.1,
    use_raw=False,
    n_perms=100,
    verbose=True
)

# --- Save results with FDR correction
res = adata_pair.uns["liana_res"].copy()

# ADDED: FDR correction for multiple testing
print("Applying FDR correction...")
_, res['qval'], _, _ = multipletests(res['magnitude_rank'], method='fdr_bh')

out_path = os.path.join(output_dir, f"{job_name}_results.csv")
res.to_csv(out_path, index=False)

# OLD (commented out): sig = res[res["magnitude_rank"] < 0.05]
# NEW: Use FDR-corrected q-values
sig = res[res["qval"] < 0.05]

print(f"✓ Saved {{out_path}}")
print(f"  Total interactions: {{len(res):,}}")
print(f"  Significant (raw p<0.05): {{(res['magnitude_rank'] < 0.05).sum():,}}")
print(f"  Significant (FDR q<0.05): {{len(sig):,}}")
'''
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    slurm_content = f'''#!/bin/bash
#SBATCH --job-name={job_name}_age
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

print(f"\n✓ Submitted {len(pairs)} jobs with age bins!")
print(f"Results will be in: {output_dir}")
