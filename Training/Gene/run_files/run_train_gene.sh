#!/bin/bash

#SBATCH --account=ai4bio2024
#SBATCH --job-name=gene_training
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

#SBATCH --output="run_output/output_train.log"
#SBATCH --error="run_output/error_train.log"

# training 
python3 train.py