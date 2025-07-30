#!/bin/bash

#SBATCH --account=ai4bio2024
#SBATCH --job-name=gene_gen_dataset
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=9:00:00

#SBATCH --output="run_output/output.log"
#SBATCH --error="run_output/error.log"

# training 
python3 generate_dataset.py 500 0.0004
python3 generate_dataset.py 200 0.0004
python3 generate_dataset.py 100 0.0004

python3 generate_dataset.py 500 0.001
python3 generate_dataset.py 200 0.001
python3 generate_dataset.py 100 0.001

python3 generate_dataset.py 500 0.0001
python3 generate_dataset.py 200 0.0001
python3 generate_dataset.py 100 0.0001
