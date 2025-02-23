#!/bin/bash

#SBATCH --account=ai4bio2024
#SBATCH --job-name=copy_number_train
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G

#SBATCH --output="run_output/output.log"
#SBATCH --error="run_output/error.log"

# training 
python3 train.py --epoch 5 --n_folds 10 --model_list GCN --dataset_list MUTAG --readout_list avg --n_agg_layer 2 --agg_hidden 32
