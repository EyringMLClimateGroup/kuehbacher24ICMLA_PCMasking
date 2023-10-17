#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=90
#SBATCH --gpus=1
#SBATCH --mem=0
#SBATCH --prefer=a100_80
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --account=bd1179
#SBATCH --mail-type=END
#SBATCH --output=../output_castle/training_28_custom_mirrored_functional/plots_offline_evaluation/run_2/shap/slurm_logs/%x_slurm_%j.out
#SBATCH --error=../output_castle/training_28_custom_mirrored_functional/plots_offline_evaluation/run_2/shap/slurm_logs/%x_error_slurm_%j.out

display_help() {
  echo ""
  echo "SLURM batch script for computing CASTLE shapley values."
  echo ""
  echo "Usage: sbatch -J job_name compute_castle_shapley.sh -c config.yml -o outputs_file.txt -x var_index -m outputs_map.txt -p plot_directory -t n_time -s n_samples -e metric [-j job_name]"
  echo ""
  echo " Options:"
  echo " -c    YAML configuration file for CASTLE network."
  echo " -o    Text file specifying output variable networks (.txt)."
  echo " -x    Index of the output variable in outputs_file.txt for which to compute the Shapley values (int)."
  echo " -m    Text file specifying the mapping between variable names and saved network names (.txt)."
  echo " -p    Output directory for shapley dictionaries and plots."
  echo " -t    Number of time samples to select from the data."
  echo " -s    Number of samples to be used for shapley computation."
  echo " -e    Metric to be used on shapley values. Can be one of ['mean', 'abs_mean', 'abs_mean_sign']."
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
found_o=0
found_x=0
found_m=0
found_p=0
found_t=0
found_s=0
found_e=0
found_j=0

# Parse options
while getopts "c:o:x:m:p:t:s:e:j:h" opt; do
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
  o)
    found_o=1
    if [[ $OPTARG == *.txt ]]; then
      OUTPUTS=$OPTARG
    else
      echo -e "\nError: Invalid value for option -o (output variable networks). Must be .txt file."
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
  m)
    found_m=1
    if [[ $OPTARG == *.txt ]]; then
      MAP=$OPTARG
    else
      echo -e "\nError: Invalid value for option -m (variables to networks mapping file). Must be .txt file."
      error_exit
    fi
    ;;
  p)
    found_p=1
    PLOT_DIR=$OPTARG
    ;;
  t)
    found_t=1
    re='^[+-]?[0-9]+$'
    if [[ $OPTARG =~ $re ]]; then
      N_TIME=$OPTARG
    else
      lower_input=$(echo "$OPTARG" | tr '[:upper:]' '[:lower:]')
      if [[ $lower_input == "false" || $lower_input == "f" ]]; then
        N_TIME="False"
      else
        echo -e "\nError: Invalid value for option -t (n_time). Must be an integer or False."
        error_exit
      fi
    fi
    ;;
  s)
    found_s=1
    re='^[+-]?[0-9]+$'
    if [[ $OPTARG =~ $re ]]; then
      N_SAMPLES=$OPTARG
    else
      echo -e "\nError: Invalid value for option -s (n_samples). Must be an integer."
      error_exit
    fi
    ;;
  e)
    found_e=1
    METRIC=$OPTARG
    ;;
  j)
    found_j=1
    JOB_NAME=$OPTARG
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
elif ((found_o == 0)); then
  echo -e "\nError: Failed to provide output variables networks .txt file.\n"
  error_exit
elif ((found_x == 0)); then
  echo -e "\nError: Failed to provide value for output variable index.\n"
  error_exit
elif ((found_m == 0)); then
  echo -e "\nError: Failed to provide output variables mapping .txt file.\n"
  error_exit
elif ((found_p == 0)); then
  echo -e "\nError: Failed to provide output directory for plots and shapleu dictionaries.\n"
  error_exit
elif ((found_t == 0)); then
  echo -e "\nError: Failed to provide value for n_time.\n"
  error_exit
elif ((found_s == 0)); then
  echo -e "\nError: Failed to provide value for n_samples.\n"
  error_exit
elif ((found_e == 0)); then
  echo -e "\nError: Failed to provide value for metric.\n"
  error_exit
fi

if ((found_j == 0)); then
  JOB_NAME="compute_castle_shap"
fi

##################################
# Start computing shapley values #
##################################

PROJECT_ROOT="$(dirname "${PWD}")"

echo "Start time: "$(date)

conda run --cwd "$PROJECT_ROOT" -n tensorflow_env python -u -m castle_offline_evaluation.main_castle_shapley -c "$CONFIG" -o "$OUTPUTS" -x "$VAR_INDEX" -m "$MAP" -p "$PLOT_DIR" -t "$N_TIME" -s "$N_SAMPLES" -e "$METRIC" >"../output_castle/training_28_custom_mirrored_functional/plots_offline_evaluation/run_2/shap/slurm_logs/${JOB_NAME}_python_${SLURM_JOB_ID}.out"
