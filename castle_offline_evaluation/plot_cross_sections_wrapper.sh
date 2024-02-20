#!/bin/bash

##################
# Set parameters #
##################

PROJECT_ROOT="$(dirname "${PWD}")"

TRAINING_DIR="${PROJECT_ROOT}/output_castle/training_72_gumbel_softmax_single_output_crf"
JOB_NAME="plots_cross_section_72_gumbel_softmax_single_output_crf"
HPC="jsc" # jsc, dkrz

CONFIG="${TRAINING_DIR}/cfg_gumbel_softmax_single_output.yml"


PLOT_DIR="${TRAINING_DIR}/plots_offline_evaluation/plots_cross_section"
mkdir -p "$PLOT_DIR"
SLURM_LOG_DIR="${PLOT_DIR}/slurm_logs"
mkdir -p "$SLURM_LOG_DIR"

################
# Help Display #
################

display_help() {
  echo ""
  echo "Wrapper for SLURM batch script for plotting cross sections."
  echo ""
  echo "Usage: ./plot_cross_sections_wrapper.sh -h"
  echo ""
  echo " Options:"
  echo " -h    Print this help."
  echo ""
  echo ""
  echo "The following parameters specifying input files and the output directory must be set in the script: "
  echo "    TRAINING_DIR  The directory containing the config file and inputs, outputs lists and mappings."
  echo "    JOB_NAME      SLURM job name."
  echo "    CONFIG        Name of YAML configuration file for the network."
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
  BATCH_SCRIPT="plot_double_yz_sbatch_jsc.sh"
elif [[ $HPC == "dkrz" ]]; then
  echo -e "\nNo batch script yet for DKRZ."
  error_exit
else
  echo -e "\nUnknown HPC ${HPC}."
  error_exit
fi

####################
# Start SLURM jobs #
####################

echo -e "\n\n--- Start plotting cross sections with\n "
echo -e " Directory:          ${TRAINING_DIR}\n"
echo -e " Output directory:   ${PLOT_DIR}\n"
echo -e " Config:             ${CONFIG}\n\n---\n"
want_to_continue

slurm_o="${SLURM_LOG_DIR}/%x_slurm_%j.out"
slurm_e="${SLURM_LOG_DIR}/%x_error_slurm_%j.out"

echo -e "\nStarting job ${JOB_NAME}"
sbatch -J "$JOB_NAME" --output "$slurm_o" --error "$slurm_e" "$BATCH_SCRIPT" -c "$CONFIG" -p "$PLOT_DIR" -l "$SLURM_LOG_DIR" -j "$JOB_NAME"

echo ""