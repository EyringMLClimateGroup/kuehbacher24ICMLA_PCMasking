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
while getopts "h" opt; do
  case ${opt} in
  h)
    display_help
    exit 0
    ;;
  esac
done
shift "$(($OPTIND - 1))"

# Set tuning parameters
lambda_prediction=(2 4 8 10)
lambda_sparsity=(0.1 0.5 1.0)

# Set base directory for config files and inputs/outputs list files
base_dir="output_castle/manual_tuning_tphystnd_691.39"
tuning_model="castle_adapted_big_notears"
in="${base_dir}/inputs_list.txt"
out="${base_dir}/outputs_list.txt"

# Index for network to be tuned: variable tphystnd-691.39
idx="20-20"

for p in "${lambda_prediction[@]}"; do
  for s in "${lambda_sparsity[@]}"; do
    dir="lambda_pred_${p}-lambda_sparsity_${s}"
    job_name="manual_tuning_tphystnd_691_adapted_big_lambda_pred_${p}-lambda_sparsity_${s}"

    log_dir="${base_dir}/${tuning_model}/${dir}"
    cfg="${log_dir}/cfg_castle_adapted.yml"

    slurm_o="${log_dir}/%x_slurm_%j.out"
    slurm_e="${log_dir}/%x_error_slurm_%j.out"

    echo -e "\nSubmitting job ${job_name}"
    sbatch --job-name "$job_name" --output "$slurm_o" --error "$slurm_e" manual_tuning_castle_batch.sh -c "$cfg" -i "$in" -o "$out" -x "$idx" -p "$log_dir" -j "$job_name"
  done
done
echo ""
