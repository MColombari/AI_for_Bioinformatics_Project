#!/bin/bash

#SBATCH --account=ai4bio2024
#SBATCH --job-name=gene_testing
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

#SBATCH --output="output_test.log"
#SBATCH --error="error_test.log"

# training 
python3 run.py