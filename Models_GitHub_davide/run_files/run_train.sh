#!/bin/bash

#SBATCH --account=ai4bio2024
#SBATCH --job-name=copy_number_train
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
<<<<<<< HEAD
#SBATCH --mem-per-cpu=16
=======
>>>>>>> 731c396f8b463a753fcabd0637b7936a50b70355

#SBATCH --output="run_output/output.log"
#SBATCH --error="run_output/error.log"

# training 
<<<<<<< HEAD
python3 train.py --epoch 5 --n_folds 5 --model_list GCN --dataset_list COPY_NUMBER --readout_list avg --n_agg_layer 2 --agg_hidden 32
=======
python3 script_generation_dataset.py
>>>>>>> 731c396f8b463a753fcabd0637b7936a50b70355
