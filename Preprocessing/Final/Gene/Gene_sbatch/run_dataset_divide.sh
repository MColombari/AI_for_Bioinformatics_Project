#!/bin/bash

#SBATCH --account=ai4bio2024
#SBATCH --job-name=dataset_divide
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=270396@studenti.unimore.it
#SBATCH --output="output_divide_dataset.log"
#SBATCH --error="error_divide_dataset.log"

# training
source /homes/mcolombari/AI_for_Bioinformatics_Project/env/training_env/bin/activate
python3 divide_dataset.py