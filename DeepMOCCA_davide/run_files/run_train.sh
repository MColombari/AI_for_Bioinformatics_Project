#!/bin/bash

#SBATCH --account=ai4bio2024
#SBATCH --job-name=gene_exp_train
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00

#SBATCH --output="run_output/output_train.log"
#SBATCH --error="run_output/error_train.log"

# training
source /homes/dlupo/Progetto_BioInformatics/AI_for_Bioinformatics_Project/env/bin/activate
python3 /homes/dlupo/Progetto_BioInformatics/AI_for_Bioinformatics_Project/DeepMOCCA_davide/deepmocca/main.py --in-file data --in-file example_file.txt --cancer-type-flag 20 --anatomical-part-flag 36
