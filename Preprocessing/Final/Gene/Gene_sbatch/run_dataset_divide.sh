#!/bin/bash

#SBATCH --account=ai4bio2024
#SBATCH --job-name=gene_variance
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

#SBATCH --output="output_divide_dataset.log"
#SBATCH --error="error_divide_dataset.log"

# training 
python3 divide_dataset.py