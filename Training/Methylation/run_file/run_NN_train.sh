#!/bin/bash

#SBATCH --account=ai4bio2024
#SBATCH --job-name=top1_testing
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

#SBATCH --output="run_output/output_test1.log"
#SBATCH --error="run_output/error_test1.log"

# training 
python3 Training/Methylation/train.py