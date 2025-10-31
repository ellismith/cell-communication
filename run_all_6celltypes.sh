#!/bin/bash
#SBATCH --job-name=cellchat_6types
#SBATCH --output=cellchat_6types_%j.out
#SBATCH --error=cellchat_6types_%j.err
#SBATCH --partition=highmem
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=1000G

source ~/.bashrc
conda activate cellchat_env
python cellchat_all_6celltypes.py
