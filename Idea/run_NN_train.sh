#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=top1_testing
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

#SBATCH --output="run_output/output_train_NN_Ensemble.log"
#SBATCH --error="run_output/error_train_NN_Ensemble.log"

# training 
python3 ensemble_NN.py