#!/bin/bash
#SBATCH --job-name=whole_region_regression
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/easmit31/cell_cell/regression_jobs_whole_region/%x_%j.out

source activate cellchat_env

python /scratch/easmit31/cell_cell/run_age_regression_with_threshold_v2.py \
    --region $REGION \
    --threshold $THRESHOLD \
    --min_age 1.0 \
    --min_animals 10
