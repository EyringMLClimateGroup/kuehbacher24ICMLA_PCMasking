#!/bin/bash

##################
# Set parameters #
##################

PROJECT_ROOT="$(dirname "${PWD}")"

TRAINING_DIR="${PROJECT_ROOT}/output_castle/training_71_gumbel_softmax_single_output_crf"
JOB_NAME="shap_71_gumbel_softmax_single_output_crf"
HPC="jsc" # jsc, dkrz

CONFIG="${TRAINING_DIR}/cfg_gumbel_softmax_single_output.yml"
INPUTS="${TRAINING_DIR}/inputs_list.txt"
OUTPUTS="${TRAINING_DIR}/outputs_list.txt"
MAP="${TRAINING_DIR}/outputs_map.txt"

PLOT_DIR="${TRAINING_DIR}/plots_offline_evaluation/shap"
mkdir -p "$PLOT_DIR"
SLURM_LOG_DIR="${PLOT_DIR}/slurm_logs"
mkdir -p "$SLURM_LOG_DIR"

N_TIME="False"
N_SAMPLES=1000
METRIC="all"

################
# Help Display #
################

display_help() {
  echo ""
  echo "Wrapper for SLURM batch script for computing CASTLE shapley values."
  echo ""
  echo "Usage: ./compute_castle_shapley_wrapper.sh -h"
  echo ""
  echo " Options:"
  echo " -h    Print this help."
  echo ""
  echo ""
  echo "The following parameters specifying input files and the output directory must be set in the script: "
  echo "    TRAINING_DIR  The directory containing the config file and inputs, outputs lists and mappings."
  echo "    JOB_NAME      SLURM job name."
  echo "    CONFIG        Name of YAML configuration file for the network."
  echo "    INPUTS        Name of network inputs list file (.txt)."
  echo "    OUTPUTS       Name of network outputs list file (.txt)."
  echo "    MAP           Name of network outputs mapping file (.txt)."
  echo "    PLOT_DIR      Output directory for shapley dictionaries and plots."
  echo ""
  echo "The following parameters must be set for SHAP computation: "
  echo "    N_TIME       Number of time samples to select from the data (int, False). Use N_TIME=False if all data should be selected."
  echo "    N_SAMPLES    Number of samples to be used for shapley computation (int)."
  echo "    METRIC       Metric to be used on shapley values. Can be one of ['mean', 'abs_mean', 'abs_mean_sign', 'all', 'none']."
  echo ""
}

error_exit() {
  echo -e "Exiting script.\n"
  exit 1
}

graceful_exit() {
  echo -e "Exiting script.\n"
  exit 0
}

want_to_continue() {
  echo ""
  read -r -e -p "Do you want to continue? [y]/n: " input
  answer=${input:-"y"}

  if [[ $answer == "y" ]]; then
    :
  elif [[ $answer == "n" ]]; then
    graceful_exit
  else
    echo "Unknown value."
    error_exit
  fi
}

# Parse options
while getopts "h" opt; do
  case ${opt} in
  h)
    display_help
    exit 0
    ;;
  \?)
    echo -e "\nInvalid option. Use option -h for help."
    error_exit
    ;;
  esac
done
shift "$(($OPTIND - 1))"

################
# Batch script #
################
if [[ $HPC == "jsc" ]]; then
  BATCH_SCRIPT="compute_castle_shapley_jsc.sh"
elif [[ $HPC == "dkrz" ]]; then
  BATCH_SCRIPT="compute_castle_shapley_dkrz.sh"
else
  echo -e "\nUnknown HPC ${HPC}."
  error_exit
fi

####################
# Start SLURM jobs #
####################

echo -e "\n\n--- Starting SHAP computation for all output networks with\n "
echo -e " Directory:          ${TRAINING_DIR}\n"
echo -e " Config:             ${CONFIG}\n"
echo -e " Output directory:   ${PLOT_DIR}\n\n---\n"
want_to_continue

slurm_o="${SLURM_LOG_DIR}/%x_slurm_%j.out"
slurm_e="${SLURM_LOG_DIR}/%x_error_slurm_%j.out"

echo -e "\nStarting job ${JOB_NAME}."
sbatch -J "$JOB_NAME" --output "$slurm_o" --error "$slurm_e" "$BATCH_SCRIPT" -c "$CONFIG" -o "$OUTPUTS" -m "$MAP" -p "$PLOT_DIR" -t "$N_TIME" -s "$N_SAMPLES" -e "$METRIC" -l "$SLURM_LOG_DIR" -j "$JOB_NAME"

echo -e "\n--- Finished.\n "
