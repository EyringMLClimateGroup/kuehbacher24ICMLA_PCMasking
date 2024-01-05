#!/bin/bash

###########################
# Default argument values #
###########################
base_dir="output_castle/tuning_10_castle_simplified_lr0.01"
HPC="jsc" # jsc, dkrz

NN_INPUTS="${base_dir}/inputs_list.txt"
NN_OUTPUTS="${base_dir}/outputs_list.txt"
CONFIG="${base_dir}/cfg_castle_simplified.yml"

SEARCH_SPACE_CONFIG="${base_dir}/search_space_config.yml"

TUNER="GridSearch"
METRIC="val_prediction_loss"
var="tphystnd_691.39"      # tphystnd_691.39, tphystnd_820.86

PORT=8123
SEED=42
JOB_NAME="tuning_${var}_${PORT}"


# Index for network to be tuned:
# tphystnd_691.39 20, tphystnd_820.86 22
if [[ $var == "tphystnd_691.39" ]]; then
  VAR_INDEX=20
elif [[ $var == "tphystnd_820.86" ]]; then
  VAR_INDEX=22
else
  echo -e "\nIndex for variable ${var} not known. Exiting script.\n"
  exit 1
fi

log_dir="${base_dir}/slurm_logs"
mkdir -p "$log_dir"

slurm_o="${log_dir}/%x_slurm_%j.out"
slurm_e="${log_dir}/%x_error_slurm_%j.out"


display_help() {
  echo ""
  echo "Wrapper to call CASTLE tuning batch script tune_castle_split_nodes_batch_dkrz/jsc.sh."
  echo ""
  echo "Set tuning configuration in this script."
  echo ""
  echo "Usage: ./tune_castle_wrapper.sh [-h]"
  echo ""
  echo " Options:"
  echo " -h    Print this help."
  echo ""
}

error_exit() {
  echo "Invalid option. See option -h for help on how to run this script."
  echo -e "Exiting script.\n"
  exit 1
}

while getopts "h" opt; do
  case ${opt} in
  h)
    display_help
    exit 0
    ;;
  \?)
    error_exit
    ;;
  esac
done
shift "$(($OPTIND - 1))"

################
# Batch script #
################
if [[ $HPC == "jsc" ]]; then
  BATCH_SCRIPT="tune_castle_split_nodes_batch_jsc.sh"
elif [[ $HPC == "dkrz" ]]; then
  BATCH_SCRIPT="tune_castle_split_nodes_batch_dkrz.sh"
else
  echo -e "\nUnknown HPC ${HPC}."
  error_exit
fi

echo -e "\nSubmitting job ${JOB_NAME}"
echo "Index: ${VAR_INDEX}"
sbatch --job-name "$JOB_NAME" --output "$slurm_o" --error "$slurm_e" "$BATCH_SCRIPT" -c "$CONFIG" -i "$NN_INPUTS" -o "$NN_OUTPUTS" -x "$VAR_INDEX" -u "$TUNER" -p "$METRIC" -e "$SEARCH_SPACE_CONFIG" -r "$PORT" -l "$log_dir" -s "$SEED" -j "$JOB_NAME"
echo ""
