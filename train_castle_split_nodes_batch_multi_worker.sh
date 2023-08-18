#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus=4
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block  # distribution, might be better to have contiguous blocks
#SBATCH --mem=0
#SBATCH --constraint=a100_80
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --account=bd1179
#SBATCH --mail-type=END
#SBATCH --output=output_castle/training_20_mirrored_custom_multi_worker/%x_slurm.%j.out

# Job name is passed with option -J and as command line argument $6
# If you don't use option -J, set #SBATCH --job-name=castle_training

#############
# Functions #
#############

display_help() {
  echo ""
  echo "SLURM batch script for training CASTLE model for specified outputs."
  echo ""
  echo "Usage: $0 [-h] [-c config.yml] [-i inputs_list.txt] [-o outputs_list.txt] [-x output_indices] [-s seed] [-j job_name]"
  echo ""
  echo " Options:"
  echo " -c    YAML configuration file for CASTLE network."
  echo " -i    Text file with input list for CASTLE networks (.txt)."
  echo " -o    Text file with output list for CASTLE networks (.txt)."
  echo " -x    Indices of outputs to be trained in 'outputs_list.txt'. Must be a string of the form 'start-end'."
  echo " -s    Random seed. Leave out this option to not set a random seed or set value to 'NULL' or 'False'."
  echo " -j    SLURM job name."
  echo " -h    Print this help."
  echo ""
}

error_exit() {
  echo -e "Exiting script.\n"
  exit 1
}

####################
# Argument parsing #
####################

found_c=0
found_i=0
found_o=0
found_x=0
found_s=0
found_j=0

# Parse options
while getopts "c:i:o:x:s:j:h" opt; do
  case ${opt} in
  h)
    display_help
    exit 0
    ;;
  c)
    found_c=1
    if [[ $OPTARG == *.yml ]]; then
      CONFIG=$OPTARG
    else
      echo -e "\nError: Invalid value for option -c (YAML config). Must be YAML file."
      error_exit
    fi
    ;;
  i)
    found_i=1
    if [[ $OPTARG == *.txt ]]; then
      INPUTS=$OPTARG
    else
      echo -e "\nError: Invalid value for option -i (CASTLE inputs list). Must be .txt file."
      error_exit
    fi
    ;;
  o)
    found_o=1
    if [[ $OPTARG == *.txt ]]; then
      OUTPUTS=$OPTARG
    else
      echo -e "\nError: Invalid value for option -i (CASTLE outputs list). Must be .txt file."
      error_exit
    fi
    ;;
  x)
    found_x=1
    START_END_IDX=$OPTARG
    ;;
  s)
    found_s=1
    re='^[+-]?[0-9]+$'
    if [[ $OPTARG =~ $re ]]; then
      SEED=$OPTARG
    elif [[ $OPTARG == "NULL" || $OPTARG == "False" ]]; then
      SEED="False"
    else
      echo -e "\nError: Invalid value for option -s (random seed). Must be an integer or NULL/False."
      error_exit
    fi
    ;;
  j)
    found_j=1
    JOB_NAME=$OPTARG
    ;;
  :)
    echo -e "\nOption $opt requires an argument."
    error_exit_help
    ;;
  \?)
    error_exit_help
    ;;
  esac
done
shift "$(($OPTIND - 1))"

if ((found_c == 0)); then
  echo -e "\nError: Failed to provide CASTLE YAML config file.\n"
  error_exit
elif ((found_i == 0)); then
  echo -e "\nError: Failed to provide CASTLE inputs list .txt file.\n"
  error_exit
elif ((found_o == 0)); then
  echo -e "\nError: Failed to provide CASTLE outputs list .txt file.\n"
  error_exit
elif ((found_x == 0)); then
  echo -e "\nError: Failed to outputs training indices.\n"
  error_exit
fi

if ((found_s == 0)); then
  SEED="False"
fi
if ((found_j == 0)); then
  JOB_NAME="castle_training_${START_END_IDX}"
fi

##################
# Start training #
##################

# This is necessary for the workers to communicate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

set -x
cd "${SLURM_SUBMIT_DIR}" || error_exit

echo "Starting job ${JOB_NAME}: $(date)"

# srun will use gres=gpu:1 by default, but we want to use all 4 gpus
srun --gres=gpu:4 conda run -n tensorflow_env python -u main_train_castle_split_nodes.py -c "$CONFIG" -i "$INPUTS" -o "$OUTPUTS" -x "$START_END_IDX" -s "$SEED" > "output_castle/training_20_mirrored_custom_multi_worker/${JOB_NAME}_python_${SLURM_JOB_ID}.out"