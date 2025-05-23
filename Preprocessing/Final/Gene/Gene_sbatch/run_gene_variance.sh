#!/bin/bash

#SBATCH --account=ai4bio2024
#SBATCH --job-name=gene_variance
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

#SBATCH --output="output_gene_variance.log"
#SBATCH --error="error_gene_variance.log"

# training 
python3 gene_variance.py