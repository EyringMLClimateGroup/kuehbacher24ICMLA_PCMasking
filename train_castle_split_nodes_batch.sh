#!/bin/bash
#SBATCH --job-name=castle_training
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=90
#SBATCH --gpus=4
#SBATCH --mem=0
#SBATCH --constraint=a100_80
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --account=bd1179
#SBATCH --output=output_castle/training_10_mirrored/castle_training_split_slurm.%j.out

echo "Starting job: $(date)"

# Args:
#  $1 config file
#  $2 NN .txt inputs file
#  $3 NN .txt outputs file
#  $4 training indices
#  $5 random seed
conda run -n tensorflow_env python -u main_train_castle_split_nodes.py -c "$1" -i "$2" -o "$3" -x "$4" -s "$5" >"output_castle/training_10_mirrored/castle_training_split_python_$SLURM_JOB_ID.out"
