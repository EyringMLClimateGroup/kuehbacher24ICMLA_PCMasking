#!/bin/bash

###########################
# Default argument values #
###########################
# todo: extract base dir from config?
base_dir="/p/scratch/icon-a-ml/kuehbacher1/output_castle/training_106_mask_net_prediction_thresholds_spars0.001_minus4k_ref"
HPC="gpu_threshold_jsc" # jsc, dkrz, gpu_jsc

job_name_base="train_106_mask_net_pred_thresh_spars0.001_minus4k_ref"

NN_INPUTS="${base_dir}/inputs_list.txt"
NN_OUTPUTS="${base_dir}/outputs_list.txt"
OUTPUTS_MAP="${base_dir}/outputs_map.txt"
NN_CONFIG="${base_dir}/cfg_mask_net.yml"

PERCENTILE=70

NUM_NODES=17
NN_PER_NODE=4
SEED=42
LOAD_CKPT="False"
CONTINUE_TRAINING="False"
FINE_TUNE_CONFIG="/p/scratch/icon-a-ml/kuehbacher1/output_castle/training_102_pre_mask_net_spars0.001_minus4k/cfg_pre_mask_net.yml"

MAX_RUNNING_JOBS_DKRZ=20

log_dir="${base_dir}/slurm_logs"
mkdir -p "$log_dir"

#############
# Functions #
#############

display_help() {
  echo ""
  echo "Bash script wrapper for CASTLE training that splits training of model description list across multiple SLURM nodes."
  echo "Training configuration parameters can either be specified in the script directly or via command line arguments."
  echo ""
  echo "Usage: $0 [-h] [-i inputs_list.txt] [-o outputs_list.txt] [-m outputs_map.txt]  [-c config.yml] [-l load_ckp_weight] [-t continue_training] [-f fine_tune_config] [-p HPC] [-s seed]"
  echo ""
  echo " Options:"
  echo " -i    txt file with input list for CASTLE networks."
  echo "       Current value: $NN_INPUTS"
  echo ""
  echo " -o    txt file with output list for CASTLE networks."
  echo "       Current value: $NN_OUTPUTS"
  echo ""
  echo " -m    txt file with mapping for output variable identifiers for CASTLE networks."
  echo "       Current value: $OUTPUTS_MAP"
  echo ""
  echo " -c    YAML configuration file for CASTLE networks."
  echo "       Current value: $NN_CONFIG"
  echo ""
  echo " -l    Boolean ('False' 'f', 'True', 't') indicating whether to load weights from checkpoint from previous training."
  echo "       Current value: $LOAD_CKPT"
  echo ""
  echo " -t    Boolean ('False' 'f', 'True', 't') indicating whether to continue with previous training. "
  echo "       The model (including optimizer) is loaded and the learning rate is initialized with the last learning rate from previous training."
  echo "       Current value: $CONTINUE_TRAINING"
  echo ""
  echo " -f    Config file for trained PreMaskNet to load weights for fine-tuning of MaskNet. "
  echo "       Hidden and output layer weights of PreMaskNet are reloaded for MaskNet training."
  echo "       If no config file is provided, MaskNet is trained from scratch."
  echo "       Current value: $FINE_TUNE_CONFIG"
  echo ""
  echo " -p    Which HPC one is working on ('dkrz' or 'jsc')."
  echo "       Current value: $HPC"
  echo ""
  echo " -s    Random seed. Leave out this option to not set a random seed."
  echo "       Current value: $SEED"
  echo ""
  echo " -h    Print this help."
  echo ""
  exit 0
}

print_variables() {
  echo -e "\n================================================================="
  echo ""
  echo "  NN inputs file:                 $NN_INPUTS"
  echo "  NN outputs file:                $NN_OUTPUTS"
  echo "  Outputs map file:               $OUTPUTS_MAP"
  echo "  NN config file:                 $NN_CONFIG"
  echo "  Distributed training:           $DISTRIBUTED"
  echo "  Load weights from checkpoint:   $LOAD_CKPT"
  echo "  Continue training:              $CONTINUE_TRAINING"
  echo "  Fine-tuning config:             $FINE_TUNE_CONFIG"
  echo "  Random Seed:                    $SEED"
  echo ""
  echo "  Number of NNs:                  $NUM_OUTPUTS"
  echo "  Number of training nodes/jobs:  $NUM_NODES"
  echo "  Number of NNs per node/job:     $NN_PER_NODE"
  echo ""
  echo "  HPC script:                     $HPC"
  echo ""
  echo -e "=================================================================\n\n"
}

error_exit_help() {
  echo -e "\nUsage: $0 [-h] [-i inputs.txt] [-o outputs.txt] [-n nodes] [-c config.yml]"
  echo -e "\nUse option -h for help.\n"
  exit 1
}

error_exit() {
  echo -e "Exiting script.\n"
  exit 1
}

graceful_exit() {
  echo -e "Exiting script.\n"
  exit 0
}

timestamp() {
  date +"%T" # current time
}

case_counter() {
  case $1 in
  0)
    echo -e "\nError: Unknown input. Try again (2 tries left).\n"
    ;;
  1)
    echo -e "\nError: Unknown input. Try again (1 try left).\n"
    ;;
  2)
    echo -e "\nError: Unknown input.\n"
    error_exit
    ;;
  esac
}

want_to_continue() {
  echo ""
  counter=0
  while [ $counter -lt 3 ]; do
    read -r -e -p "Do you want to continue? [y]/n: " input
    answer=${input:-"y"}

    if [[ $answer == "y" ]]; then
      NUM_NODES=$TEMP
      break
    elif [[ $answer == "n" ]]; then
      graceful_exit
    else
      case_counter $counter
    fi
    counter=$(($counter + 1))
  done
}

read_distributed() {
  if [ -f "$NN_CONFIG" ]; then
    TMP=$(grep 'distribute_strategy:' $NN_CONFIG)
    TMP=${TMP//*distribute_strategy: /}
    # Remove comments if there are any
    DISTRIBUTED=${TMP%#*}
    # Remove trailing new lines and spaces
    DISTRIBUTED="${DISTRIBUTED//[$'\t\r\n\" ']/}"
  else
    echo -e "\nError: YAML configuration file does not exist.\n"
    error_exit
  fi

  if [[ $DISTRIBUTED == "" ]]; then
    DISTRIBUTED="None"
  fi
}

check_inputs_file_exists() {
  if [ -f "$NN_INPUTS" ]; then
    :
  else
    echo -e "\nError: Inputs .txt file does not exist.\n"
    error_exit
  fi
}
check_map_file_exists() {
  if [ -f "$OUTPUTS_MAP" ]; then
    NUM_OUTPUTS="$(grep -c ".*" $NN_OUTPUTS)"
  else
    echo -e "\nError: Outputs map .txt file does not exist.\n"
    error_exit
  fi
}

check_outputs_file_exists() {
  # Check if outputs .txt file exists
  if [ -f "$NN_OUTPUTS" ]; then
    :
  else
    echo -e "\nError: Outputs .txt file does not exist.\n"
    error_exit
  fi
}

set_var_ident() {
  var_line=$(head -n "$1" $NN_OUTPUTS | tail -1)
  var_line="${var_line//[$'\t\r\n ']/}"

  var_map=$(grep "$var_line" $OUTPUTS_MAP)
  var_ident=${var_map%:*}
}

set_var_ident_str() {
  # Watch out: head starts at index 1
  start=$(($1 + 1))
  end=$(($2 + 1))

  set_var_ident $start
  var_ident_str="${var_ident}"
  set_var_ident $end
  var_ident_str="${var_ident_str}-${var_ident}"

  # set variable
  VAR_IDENT_STR=$var_ident_str
}

#######################
# Check files exists #
#######################
check_outputs_file_exists
check_inputs_file_exists
check_map_file_exists

#######################################
# Read distributed strategy from YAML #
#######################################
read_distributed

NUM_NODES=$((NUM_OUTPUTS / 4))
remainder=$((NUM_OUTPUTS % 4))

if [[ $remainder -gt 0 ]]; then
  NUM_NODES=$((NUM_NODES + 1))
fi

echo -e "\n\nRunning script with the following variables:"
print_variables
###################################
# Check: Do you want to continue? #
###################################
want_to_continue

################
# Batch script #
################

if [[ $HPC == "gpu_threshold_jsc" ]]; then
  BATCH_SCRIPT="train_castle_split_nodes_batch_gpu_threshold_jsc.sh"
elif [[ $HPC == "gpu_threshold_dkrz" ]]; then
  BATCH_SCRIPT="train_castle_split_nodes_batch_gpu_threshold_dkrz.sh"
else
  echo -e "\nUnknown HPC batch script option ${HPC}."
  error_exit
fi

echo -e "\n\n$(timestamp) --- Starting SLURM jobs.\n"

#####################
# Start SLURM nodes #
#####################

for ((i = 0; i < $NUM_OUTPUTS; i += $NN_PER_NODE)); do
  END_INDEX=$(($i + $NN_PER_NODE - 1))
  # Test if we've gone too far
  END_INDEX=$(($((($NUM_OUTPUTS - 1) < $END_INDEX)) ? $(($NUM_OUTPUTS - 1)) : $END_INDEX))
  TRAIN_INDICES="$i-$END_INDEX"

  # Set variable VAR_IDENT_STR
  set_var_ident_str "$i" "$END_INDEX"
  JOB_NAME="${job_name_base}_${VAR_IDENT_STR}"

  # Check size of string (otherwise this may cause problems saving files
  if [[ ${#JOB_NAME} -gt 70 ]]; then
    JOB_NAME="training_model"
  fi

  slurm_o="${log_dir}/%x_slurm_%j.out"
  slurm_e="${log_dir}/%x_error_slurm_%j.out"

  echo -e "\nStarting batch script with output indices $TRAIN_INDICES"
  echo "Job name: ${JOB_NAME}"

  sbatch -J "$JOB_NAME" --output "$slurm_o" --error "$slurm_e" "$BATCH_SCRIPT" -c "$NN_CONFIG" -i "$NN_INPUTS" -o "$NN_OUTPUTS" -x "$TRAIN_INDICES" -r "$PERCENTILE" -l "$LOAD_CKPT" -t "$CONTINUE_TRAINING" -f "$FINE_TUNE_CONFIG" -s "$SEED" -j "$JOB_NAME" -p "$log_dir"
done

echo -e "\n$(timestamp) --- Finished starting batch scripts.\n\n"
exit 0
