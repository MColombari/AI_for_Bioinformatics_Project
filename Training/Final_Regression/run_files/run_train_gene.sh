#!/bin/bash

#SBATCH --account=ai4bio2024
#SBATCH --job-name=final_regression_training
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=270396@studenti.unimore.it

#SBATCH --output="run_output/output_train.log"
#SBATCH --error="run_output/error_train.log"

# training 
source /homes/mcolombari/AI_for_Bioinformatics_Project/env/training_env/bin/activate
python3 train.py