#!/bin/bash
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --account=icon-a-ml
#SBATCH --mail-user=birgit.kuehbacher@dlr.de
#SBATCH --mail-type=FAIL

# Job name is passed with option -J and as command line argument $6
# If you don't use option -J, set #SBATCH --job-name=castle_training

# Output streams are passed with option -e and -o
# If you don't use these options, set #SBATCH --output=output_dir/%x_slurm_%j.out and #SBATCH --error=output_dir/%x_error_slurm_%j.out

# For starting one job after another has finished, add
# #SBATCH --dependency=afterok:$JOBID1

display_help() {
  echo ""
  echo "SLURM batch script for training CASTLE model for specified outputs."
  echo ""
  echo "Usage: sbatch -J job_name --output slurm_output_logs --error slurm_error_logs train_castle_split_nodes_batch_jsc.sh -c config.yml -i inputs_list.txt -o outputs_list.txt -x output_indices -p python_output_dir [-l load_ckp_weight] [-t continue_training] [-s seed] [-j job_name]"
  echo ""
  echo " Options:"
  echo " -c    YAML configuration file for CASTLE network."
  echo " -i    Text file with input list for CASTLE networks (.txt)."
  echo " -o    Text file with output list for CASTLE networks (.txt)."
  echo " -x    Indices of outputs to be trained in 'outputs_list.txt'. Must be a string of the form 'start-end'."
  echo " -l    Boolean ('False' 'f', 'True', 't') indicating whether to load weights from checkpoint from previous training."
  echo " -t    Boolean ('False' 'f', 'True', 't') indicating whether to continue with previous training. "
  echo "       The model (including optimizer) is loaded and the learning rate is initialized with the last learning rate from previous training."
  echo " -s    Random seed. Leave out this option to not set a random seed or set value to 'NULL' or 'False'."
  echo " -p    Output directory for Python logs."
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
found_l=0
found_t=0
found_p=0

# Parse options
while getopts "c:i:o:x:s:j:l:t:p:h" opt; do
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
  l)
    found_l=1
    lower_input=$(echo "$OPTARG" | tr '[:upper:]' '[:lower:]')
    if [[ $lower_input == "true" || $lower_input == "t" ]]; then
      LOAD_CKPT="True"
    elif [[ $lower_input == "false" || $lower_input == "f" ]]; then
      LOAD_CKPT="False"
    else
      echo -e "\nError: Invalid value for option -l (load from checkpoint). Must be a boolean ('True', 't', 'False', 'f')."
      error_exit
    fi
    ;;
  t)
    found_t=1
    lower_input=$(echo "$OPTARG" | tr '[:upper:]' '[:lower:]')
    if [[ $lower_input == "true" || $lower_input == "t" ]]; then
      CONTINUE_TRAINING="True"
    elif [[ $lower_input == "false" || $lower_input == "f" ]]; then
      CONTINUE_TRAINING="False"
    else
      echo -e "\nError: Invalid value for option -t (continue training). Must be a boolean ('True', 't', 'False', 'f')."
      error_exit
    fi
    ;;
  p)
    found_p=1
    PYTHON_DIR=$OPTARG
    ;;
  :)
    echo -e "\nOption $opt requires an argument."
    echo -e "Use option -h for help."
    error_exit
    ;;
  \?)
    echo -e "\nInvalid option. Use option -h for help."
    error_exit
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
  echo -e "\nError: Failed to provide training indices.\n"
  error_exit
elif ((found_p == 0)); then
  echo -e "\nError: Failed to provide output directory for Python logs.\n"
  error_exit
fi

if ((found_s == 0)); then
  SEED="False"
fi
if ((found_j == 0)); then
  JOB_NAME="castle_training_${START_END_IDX}"
fi
if ((found_l == 0)); then
  LOAD_CKPT="False"
fi
if ((found_t == 0)); then
  CONTINUE_TRAINING="False"
fi

##################
# Start training #
##################

echo "Starting job ${JOB_NAME}: $(date)"

start_idx=$(echo $START_END_IDX | sed 's@^[^0-9]*\([0-9]\+\).*@\1@')

for gpu_index in {0..3}; do

  var_index=$((start_idx + gpu_index))

  if [ $var_index -ge 65 ]; then
    break 2
  fi

  echo -e "\nRun with GPU: ${gpu_index}\n"
  new_start_end_idx="${var_index}-${var_index}"

  conda run --no-capture-output -n kuehbacher1_py3.9_tf python -u main_train_castle_split_nodes.py -c "$CONFIG" -i "$INPUTS" -o "$OUTPUTS" -x "$new_start_end_idx" -l "$LOAD_CKPT" -t "$CONTINUE_TRAINING" -s "$SEED" -g "$gpu_index" >"${PYTHON_DIR}/${JOB_NAME}_python_${var_index}_${SLURM_JOB_ID}.out" 2>&1 &
done

wait
