#!/bin/bash
LOG_DIR="/scratch/easmit31/cell_cell/logs_corrected"
mkdir -p "$LOG_DIR"

for region in ACC CN dlPFC EC HIP IPP lCb M1 MB mdTN NAc; do
    sbatch \
        --job-name="lr_matrix_${region}" \
        --output="${LOG_DIR}/lr_matrix_${region}_%j.out" \
        --partition=general \
        --time=2:00:00 \
        --mem=64G \
        --wrap="source ~/.bashrc && conda activate cellchat_env && \
                cd /scratch/easmit31/cell_cell && \
                python build_lr_matrix.py --region ${region} --threshold 0 --min_age 1.0"
done
