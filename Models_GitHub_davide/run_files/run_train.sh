#!/bin/bash

#SBATCH --account=ai4bio2024
#SBATCH --job-name=copy_number_train
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

#SBATCH --output="run_output/output.log"
#SBATCH --error="run_output/error.log"

# training 
python3 script_generation_dataset.py