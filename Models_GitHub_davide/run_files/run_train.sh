#!/bin/bash

#SBATCH --account=ai4bio2024
#SBATCH --job-name=copy_number_train
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=32G


#SBATCH --output="run_output/output.log"
#SBATCH --error="run_output/error.log"

# training 
python3 train.py --epoch 100 --n_folds 8 --model_list GCN --dataset_list COPY_NUMBER --readout_list avg --n_agg_layer 1 --agg_hidden 32
