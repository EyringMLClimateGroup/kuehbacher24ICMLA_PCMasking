#!/bin/bash

##############################
# Set parameters from tuning #
##############################

# Set tuned parameters
lambda_prediction=(1 5 10 100 1000) #(1 2 4 8 10)
lambda_sparsity=(1.0)                     #(0.1 0.5 1.0)

PROJECT_ROOT="$(dirname "${PWD}")"

# Set base directory for config files and inputs/outputs list files
base_dir="${PROJECT_ROOT}/output_castle/manual_tuning_tphystnd_820.86_v4-transposed_l1_entry_scaled" # manual_tuning_tphystnd_820.86_v3, manual_tuning_tphystnd_691.39_v3
var="tphystnd_820.86" # tphystnd_691.39, tphystnd_820.86
job_prefix="v4-transposed_l1_entry_scaled"

tuning_models=("castle_adapted_big_notears")                                    #("castle_adapted_small_dagma" "castle_adapted_big_dagma" "castle_adapted_small_notears" "castle_adapted_big_notears")
in="${base_dir}/inputs_list.txt"
out="${base_dir}/outputs_list.txt"
map="${base_dir}/outputs_map.txt"

# Index for network to be tuned: variable tphystnd-691.39
# Index for network to be tuned:
# tphystnd_691.39 20, tphystnd_820.86 22
if [[ $var == "tphystnd_691.39" ]]; then
  idx="20"
elif [[ $var == "tphystnd_820.86" ]]; then
  idx="22"
else
  echo -e "\nIndex for variable ${var} not known. Exiting script.\n"
  exit 1
fi

#######################################
# Set parameters for SHAP computation #
#######################################

n_time="False"
n_samples=1000
metric="all"

################
# Help Display #
################

display_help() {
  echo ""
  echo "Wrapper to compute SHAP values for CASTLE manual tuning results."
  echo ""
  echo "Set tuned parameters (prediction and sparsity loss weighting coefficient) as well as"
  echo "the base directory for config files and parameters for SHAP computation in this script."
  echo ""
  echo "Usage: ./manual_tuning_compute_castle_shapley_wrapper.sh [-h]"
  echo ""
  echo " Options:"
  echo " -h    Print this help."
  echo ""
}

error_exit() {
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
    echo -e "\nInvalid option. Use option -h for help."
    error_exit
    ;;
  esac
done
shift "$(($OPTIND - 1))"

##############
# Start jobs #
##############

echo -e "\n\nBase directory: ${base_dir}\n"

for tuning_model in "${tuning_models[@]}"; do
  for p in "${lambda_prediction[@]}"; do
    for s in "${lambda_sparsity[@]}"; do
      dir="lambda_pred_${p}-lambda_sparsity_${s}"
      job_name="shap_${job_prefix}_${var}_${tuning_model}_la_pred-${p}_la_sparsity-${s}"

      log_dir="${base_dir}/${tuning_model}/${dir}"
      plot_dir="${base_dir}/evaluation/${tuning_model}/${dir}/shap"

      mkdir -p "$plot_dir"
      cfg="${log_dir}/cfg_castle_adapted.yml"

      slurm_o="${plot_dir}/%x_slurm_%j.out"
      slurm_e="${plot_dir}/%x_error_slurm_%j.out"

      echo -e "\nSubmitting job ${job_name}"
      echo -e "Index: ${idx}"
      sbatch --job-name "$job_name" --output "$slurm_o" --error "$slurm_e" manual_tuning_compute_castle_shapley.sh -c "$cfg" -o "$out" -x $idx -m "$map" -p "$plot_dir" -t "$n_time" -s "$n_samples" -e "$metric" -l "$plot_dir" -j "$job_name"
    done
  done
done
echo ""
