#!/usr/bin/env python3
"""
Generate and submit separate jobs for each cell type pair.
- Groups by BOTH cell_type AND region to preserve region info in results
- Uses *_converted.h5ad
- Uses inner gene intersection
- Results will have separate rows for each region
"""
import itertools
import os

base_path = "/scratch/easmit31/GRN_copy/scenic/h5ad_adult_files/"
output_dir = "/scratch/easmit31/cell_cell/results/pairwise"
os.makedirs("pairwise_jobs", exist_ok=True)

cell_types = {
    'Astrocyte': 'adult_grn_astrocytes_selected_converted.h5ad',
    'GABA': 'adult_grn_gaba_selected_converted.h5ad',
    'Glutamatergic': 'adult_grn_glut_selected_converted.h5ad',
    'MSN': 'adult_grn_MSN_selected_converted.h5ad',
    'OPC_Oligo': 'adult_grn_opc-olig_selected_converted.h5ad',
    'Microglia': 'adult_grn_microglia_selected_converted.h5ad'
}

pairs = list(itertools.combinations(cell_types.keys(), 2))
print(f"Creating {len(pairs)} jobs...")

for type1, type2 in pairs:
    job_name = f"{type1}_{type2}"
    script_path = f"pairwise_jobs/{job_name}.py"
    
    script_content = f'''#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import liana as li
import os

print("="*70)
print("Analyzing {type1} ↔ {type2}")
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

# --- Create combined grouping variable: cell_type + region
adata1.obs["cell_type_region"] = adata1.obs["cell_type"] + "_" + adata1.obs["region"].astype(str)
adata2.obs["cell_type_region"] = adata2.obs["cell_type"] + "_" + adata2.obs["region"].astype(str)

print(f"  {{adata1.n_obs:,}} {type1} cells, {{adata2.n_obs:,}} {type2} cells")

# --- Combine (keep intersecting genes)
adata_pair = sc.concat([adata1, adata2], join="inner")

# --- Fill NaNs and normalize
adata_pair.X[pd.isna(adata_pair.X)] = 0
sc.pp.normalize_total(adata_pair, target_sum=1e4)
sc.pp.log1p(adata_pair)

# --- Run LIANA aggregate grouping by cell_type_region (not just cell_type!)
print("Running CellChat with region info preserved...")
li.mt.rank_aggregate(
    adata_pair,
    groupby="cell_type_region",  # KEY CHANGE!
    resource_name="cellchatdb",
    expr_prop=0.1,
    use_raw=False,
    n_perms=100,
    verbose=True
)

# --- Save results - now has separate rows per region!
res = adata_pair.uns["liana_res"].copy()
out_path = os.path.join(output_dir, f"{job_name}_results.csv")
res.to_csv(out_path, index=False)

sig = res[res["magnitude_rank"] < 0.05]
print(f"✓ Saved {{out_path}} ({{len(res):,}} total, {{len(sig):,}} significant)")
'''
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    slurm_content = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=pairwise_jobs/{job_name}_%j.out
#SBATCH --error=pairwise_jobs/{job_name}_%j.err
#SBATCH --partition=htc
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

source ~/.bashrc
conda activate cellchat_env
python {script_path}
'''
    
    with open(f"pairwise_jobs/{job_name}.sh", "w") as f:
        f.write(slurm_content)
    
    os.system(f"sbatch pairwise_jobs/{job_name}.sh")
    print(f"  Submitted: {job_name}")

print(f"\\n✓ Submitted {len(pairs)} jobs!")
print(f"Results will be in: {output_dir}")
