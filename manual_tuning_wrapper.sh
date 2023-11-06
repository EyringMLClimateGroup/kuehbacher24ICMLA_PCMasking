#!/bin/bash

display_help() {
  echo ""
  echo "Wrapper to call CASTLE manual tuning batch script manual_tuning_castle_batch.sh."
  echo ""
  echo "Set tuning parameters (prediction and sparsity loss weighting coefficient)"
  echo "as well as the base directory for config files in this script."
  echo ""
  echo "Usage: ./manual_tuning_wrapper.sh [-h]"
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

# Set tuning parameters
lambda_prediction=(1 2 4 8 10)
lambda_sparsity=(0.1 0.5 1.0)

# Set base directory for config files and inputs/outputs list files
base_dir="output_castle/manual_tuning_tphystnd_691.39_v3"
trial="trial_2"
tuning_models=("castle_adapted_small_dagma" "castle_adapted_big_dagma" "castle_adapted_small_notears" "castle_adapted_big_notears")
in="${base_dir}/inputs_list.txt"
out="${base_dir}/outputs_list.txt"

# Index for network to be tuned: variable tphystnd-691.39
idx="20-20"

for tuning_model in "${tuning_models[@]}"; do
  echo -e "\n\nModel: ${tuning_model}"
  for p in "${lambda_prediction[@]}"; do
    for s in "${lambda_sparsity[@]}"; do
      dir="lambda_pred_${p}-lambda_sparsity_${s}"
      job_name="manual_tuning_tphystnd_691_${tuning_model}_lambda_pred_${p}-lambda_sparsity_${s}"

      log_dir="${base_dir}/${trial}/${tuning_model}/${dir}"
      cfg="${log_dir}/cfg_castle_adapted.yml"

      slurm_o="${log_dir}/%x_slurm_%j.out"
      slurm_e="${log_dir}/%x_error_slurm_%j.out"

      echo -e "\nSubmitting job ${job_name}"
      sbatch --job-name "$job_name" --output "$slurm_o" --error "$slurm_e" manual_tuning_castle_batch.sh -c "$cfg" -i "$in" -o "$out" -x "$idx" -p "$log_dir" -j "$job_name"
    done
  done
  echo ""
done
echo ""
