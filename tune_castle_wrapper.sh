#!/bin/bash

###########################
# Default argument values #
###########################
base_dir="output_castle/tuning_14_tphystnd_691.39_gumbel_softmax_single_output_hebo_loss"
HPC="dkrz" # jsc, dkrz

#tuning_10_tphystnd_691.39_gumbel_softmax_single_output_hebo_pred_loss
#tuning_11_tphystnd_820.86_gumbel_softmax_single_output_hebo_pred_loss
#tuning_12_tphystnd_691.39_gumbel_softmax_single_output_hebo_loss
#tuning_13_tphystnd_820.86_gumbel_softmax_single_output_hebo_loss

echo -e "\n\n-----\nTuning directory is: ${base_dir}\n"

TUNER="HEBO" # "BasicVariantGenerator", "BayesOptSearch" "HEBO"
RESTORE_DIR="${base_dir}/ray_results/tuning_GumbelSoftmaxSingleOutputModel_s42_20240116-183450"
SEED=42 # 23, 42, 36

if [[ $base_dir == *"_pred_loss"* ]]; then
  METRIC="val_prediction_loss"
elif [[ $base_dir == *"_loss"* ]]; then
  METRIC="val_loss"
else
  echo -e "\nDid not find metric pred_loss or loss in name of base directory. Exiting script.\n"
  exit 1
fi

NN_INPUTS="${base_dir}/inputs_list.txt"
NN_OUTPUTS="${base_dir}/outputs_list.txt"
CONFIG="${base_dir}/cfg_gumbel_softmax_single_output.yml"

SEARCH_SPACE_CONFIG="${base_dir}/search_space_config.yml"

if [[ $base_dir == *"tphystnd_691.39"* ]]; then
  var="tphystnd_691.39"
elif [[ $base_dir == *"tphystnd_820.86"* ]]; then
  var="tphystnd_820.86"
else
  echo -e "\nDid not find any of the variables [tphystnd_691.39, tphystnd_820.86] in name of base directory. Exiting script.\n"
  exit 1
fi

JOB_NAME="tuning_${var}_${TUNER}_${METRIC}_s${SEED}"

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
sbatch --job-name "$JOB_NAME" --output "$slurm_o" --error "$slurm_e" "$BATCH_SCRIPT" -c "$CONFIG" -i "$NN_INPUTS" -o "$NN_OUTPUTS" -x "$VAR_INDEX" -u "$TUNER" -p "$METRIC" -e "$SEARCH_SPACE_CONFIG" -l "$log_dir" -d "$RESTORE_DIR" -s "$SEED" -j "$JOB_NAME"
#echo "--job-name '$JOB_NAME' --output '$slurm_o' --error '$slurm_e' '$BATCH_SCRIPT' -c '$CONFIG' -i '$NN_INPUTS' -o '$NN_OUTPUTS' -x '$VAR_INDEX' -u '$TUNER' -p '$METRIC' -e '$SEARCH_SPACE_CONFIG' -l '$log_dir' -d '$RESTORE_DIR' -s '$SEED' -j '$JOB_NAME'"
echo -e "-----\n\n"
