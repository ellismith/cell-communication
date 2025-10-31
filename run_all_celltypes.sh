#!/bin/bash
#SBATCH --job-name=cellchat_all
#SBATCH --output=cellchat_all_%j.out
#SBATCH --error=cellchat_all_%j.err
#SBATCH --partition=htc
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

source ~/.bashrc
conda activate cellchat_env
python cellchat_all_celltypes.py
