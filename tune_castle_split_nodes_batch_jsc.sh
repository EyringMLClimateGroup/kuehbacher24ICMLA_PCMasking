#!/bin/bash
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=90
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=3:00:00
#SBATCH --account=icon-a-ml
#SBATCH --mail-user=birgit.kuehbacher@dlr.de
#SBATCH --mail-type=END

# Job name is passed with option -J and as command line argument $6
# If you don't use option -J, set #SBATCH --job-name=castle_training

# Output streams are passed with option -e and -o
# If you don't use these options, set #SBATCH --output=output_dir/%x_slurm_%j.out and #SBATCH --error=output_dir/%x_error_slurm_%j.out

# For starting one job after another has finished, add
# #SBATCH --dependency=afterok:$JOBID1

display_help() {
  echo ""
  echo "SLURM batch script for tuning CASTLE model for specified outputs."
  echo ""
  echo "Usage: sbatch -J job_name --output slurm_output_logs --error slurm_error_logs tune_castle_split_nodes_batch_jsc.sh -c config.yml -i inputs_list.txt -o outputs_list.txt -x var_index -u tuner -p metric -e search_space.yml -l python_log_dir [-d restore_dir] [-s seed] [-j job_name]"
  echo ""
  echo " Options:"
  echo " -c    YAML configuration file for CASTLE network."
  echo " -i    Text file with input list for CASTLE networks (.txt)."
  echo " -o    Text file with output list for CASTLE networks (.txt)."
  echo " -x    Index of the output variable in outputs_file.txt for which to compute the Shapley values (int)."
  echo " -u    Tuning algorithm to be used (e.g. TPE, Random, Hyperband, GP)."
  echo " -p    Tuning metric used to measure performance (eg. val_loss, val_prediction_loss)."
  echo " -e    YAML configuration file for tuning search space."
  echo " -l    Output directory for Python logs."
  echo " -d    Restore directory for continuing previous experiment (str)."
  echo " -s    Random seed. Leave out this option to not set a random seed or set value to 'NULL' or 'False'."
  echo " -j    SLURM job name."
  echo " -h    Print this help."
  echo ""
}

error_exit() {
  echo -e "Exiting script.\n"
  exit 1
}

############################
# Read and check arguments #
############################

found_c=0
found_i=0
found_o=0
found_x=0
found_s=0
found_j=0
found_u=0
found_p=0
found_e=0
found_l=0
found_d=0

# Parse options
while getopts "c:i:o:x:s:j:u:p:l:e:d:h" opt; do
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
      echo -e "\nError: Invalid value for option -c (neural network YAML config). Must be YAML file."
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
    re='^[+-]?[0-9]+$'
    if [[ $OPTARG =~ $re ]]; then
      VAR_INDEX=$OPTARG
    else
      echo -e "\nError: Invalid value for option -x (var_index). Must be an integer."
      error_exit
    fi
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
  p)
    found_p=1
    METRIC=$OPTARG
    ;;
  l)
    found_l=1
    PYTHON_DIR=$OPTARG
    ;;
  e)
    found_e=1
    if [[ $OPTARG == *.yml ]]; then
      SEARCH_SPACE_CONFIG=$OPTARG
    else
      echo -e "\nError: Invalid value for option -e (search space YAML config). Must be YAML file."
      error_exit
    fi
    ;;
  u)
    found_u=1
    TUNER=$OPTARG
    ;;
  d)
    found_d=1
    RESTORE_DIR=$OPTARG
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
  echo -e "\nError: Failed to provide value for output variable index.\n"
  error_exit
elif ((found_l == 0)); then
  echo -e "\nError: Failed to provide output directory for Python logs.\n"
  error_exit
elif ((found_u == 0)); then
  echo -e "\nError: Failed to provide tuning algorithm.\n"
  error_exit
elif ((found_p == 0)); then
  echo -e "\nError: Failed to provide tuning metric.\n"
  error_exit
elif ((found_e == 0)); then
  echo -e "\nError: Failed to provide YAML file for tuning search space configuration.\n"
  error_exit
fi

if ((found_s == 0)); then
  SEED="False"
fi
if ((found_j == 0)); then
  JOB_NAME="castle_tuning_${PORT}"
fi
if ((found_d == 0)); then
  RESTORE_DIR=""
fi

##################
# Start training #
##################

echo "Starting job ${JOB_NAME}: $(date)"

conda run --no-capture-output -n kuehbacher1_py3.9_tf python -u main_train_castle_split_nodes_tuning.py -c "$CONFIG" -i "$INPUTS" -o "$OUTPUTS" -x "$VAR_INDEX" -u "$TUNER" -p "$METRIC" -e "$SEARCH_SPACE_CONFIG" -d "$RESTORE_DIR" -s "$SEED" >"${PYTHON_DIR}/${JOB_NAME}_python_${SLURM_JOB_ID}.out"
