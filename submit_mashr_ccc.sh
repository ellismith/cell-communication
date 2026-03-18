#!/bin/bash
# Usage: bash submit_mashr_ccc.sh --source_ct Astrocyte [--louvain] [--nan_filter 0.55]

MODE=$1
CT=$2
shift 2
EXTRA="$@"

if [ "$MODE" == "--source_ct" ]; then
    LABEL="${CT}_sender"
elif [ "$MODE" == "--target_ct" ]; then
    LABEL="${CT}_receiver"
else
    echo "Usage: bash submit_mashr_ccc.sh --source_ct|--target_ct CellType [--louvain] [--nan_filter 0.55]"
    exit 1
fi

# Append extra args to label for log naming
EXTRA_TAG=$(echo "$EXTRA" | tr ' ' '_' | tr -d '-.')
LOG_DIR="/scratch/easmit31/cell_cell/results/mashr/${LABEL}${EXTRA_TAG}"
mkdir -p ${LOG_DIR}

sbatch << SLURM
#!/bin/bash
#SBATCH --job-name=mashr_${LABEL}
#SBATCH --output=${LOG_DIR}/mashr_${LABEL}_%j.out
#SBATCH --error=${LOG_DIR}/mashr_${LABEL}_%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --partition=highmem

source ~/.bashrc
conda activate mashr_env

Rscript /scratch/easmit31/cell_cell/run_mashr_ccc.R ${MODE} ${CT} ${EXTRA}
SLURM

echo "Submitted mashr job for ${LABEL} ${EXTRA}"
