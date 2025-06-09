#!/bin/bash

#SBATCH --account=ai4bio2024
#SBATCH --job-name=gene_edge_gen
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G

#SBATCH --output="run_output/output_gen_edge.log"
#SBATCH --error="run_output/error_gen_edge.log"

# training 
python3 generate_edge_file.py 900