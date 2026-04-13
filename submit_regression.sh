#!/bin/bash
LOG_DIR="/scratch/easmit31/cell_cell/logs_corrected"
mkdir -p "$LOG_DIR"

for region in CN dlPFC EC HIP IPP lCb M1 MB mdTN NAc; do
    sbatch \
        --job-name="regression_${region}" \
        --output="${LOG_DIR}/regression_${region}_%j.out" \
        --partition=general \
        --time=2:00:00 \
        --mem=64G \
        --wrap="source ~/.bashrc && conda activate cellchat_env && \
                cd /scratch/easmit31/cell_cell && \
                python run_age_sex_regression.py --region ${region}"
done

# ACC with 8H2 excluded
sbatch \
    --job-name="regression_ACC" \
    --output="${LOG_DIR}/regression_ACC_%j.out" \
    --partition=general \
    --time=2:00:00 \
    --mem=64G \
    --wrap="source ~/.bashrc && conda activate cellchat_env && \
            cd /scratch/easmit31/cell_cell && \
            python run_age_sex_regression.py --region ACC --exclude_animals 8H2"
