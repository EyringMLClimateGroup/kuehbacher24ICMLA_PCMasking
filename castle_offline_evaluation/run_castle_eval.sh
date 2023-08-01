#!/bin/bash
#SBATCH --job-name=castle_eval
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3000
#SBATCH --time=0:10:00
#SBATCH --account=bd1179
#SBATCH --output=logs/log_test_batch_slurm.%j.out

PROJECT_ROOT="$(dirname ${PWD})"

echo "Starting job: "`date`

conda run --cwd $PROJECT_ROOT -n tensorflow_env python -u -m castle_offline_evaluation.test_batch  > "logs/log_test_batch_python_$SLURM_JOB_ID.out"

