#!/bin/bash

#SBATCH --account=ai4bio2024
#SBATCH --job-name=gene_testing
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00

#SBATCH --output="run_output/output_test.log"
#SBATCH --error="run_output/error_test.log"

# training 
python3 test.py