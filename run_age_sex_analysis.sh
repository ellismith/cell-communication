#!/bin/bash
#SBATCH --job-name=cellchat_age_sex
#SBATCH --output=cellchat_age_sex_%j.out
#SBATCH --error=cellchat_age_sex_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

source ~/.bashrc
conda activate cellchat_env
python analyze_age_sex_effects.py
