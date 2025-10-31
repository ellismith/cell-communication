#!/bin/bash
#SBATCH --job-name=cellchat_9regions
#SBATCH --output=cellchat_9regions_%j.out
#SBATCH --error=cellchat_9regions_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

source ~/.bashrc
conda activate cellchat_env
python analyze_regional_effects.py
