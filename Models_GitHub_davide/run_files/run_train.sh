#!/bin/bash

#SBATCH --account=ai4bio2024
#SBATCH --job-name=copy_number_train
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00

#SBATCH --output="run_output/output.log"
#SBATCH --error="run_output/error.log"

# training 
python3 train.py --epoch 100 --n_folds 10 --model_list GAT --dataset_list COPY_NUMBER --readout_list avg --n_agg_layer 1 --agg_hidden 32
