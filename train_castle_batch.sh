#!/bin/bash
#SBATCH --job-name=castle_test_training
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=90
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --account=bd1179
#SBATCH --output=castle_test_training.%j.out

conda run -n tensorflow_env python main_train_castle.py