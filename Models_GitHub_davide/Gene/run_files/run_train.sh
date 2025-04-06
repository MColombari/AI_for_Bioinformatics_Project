#!/bin/bash

#SBATCH --account=ai4bio2024
#SBATCH --job-name=gene_exp_train
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00

#SBATCH --output="run_output/output_train.log"
#SBATCH --error="run_output/error_train.log"

# training
cd ..
python3 train.py --epoch 100 --n_folds 10 --model_list GCN --dataset_list GENE_EXP_100_001 GENE_EXP_100_0001 GENE_EXP_100_0004 --readout_list avg --n_agg_layer 1 --agg_hidden 32
python3 train.py --epoch 100 --n_folds 10 --model_list GCN --dataset_list GENE_EXP_200_001 GENE_EXP_200_0001 GENE_EXP_200_0004 --readout_list avg --n_agg_layer 1 --agg_hidden 32
python3 train.py --epoch 100 --n_folds 10 --model_list GCN --dataset_list GENE_EXP_500_001 GENE_EXP_500_0001 GENE_EXP_500_0004 --readout_list avg --n_agg_layer 1 --agg_hidden 32
