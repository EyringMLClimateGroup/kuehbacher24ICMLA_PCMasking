#!/bin/bash
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=90
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=3:00:00
#SBATCH --account=icon-a-ml
#SBATCH --mail-user=birgit.kuehbacher@dlr.de
#SBATCH --mail-type=END

# Job name is passed with option -J and as command line argument $6
# If you don't use option -J, set #SBATCH --job-name=castle_training

# Output streams are passed with option -e and -o
# If you don't use these options, set #SBATCH --output=output_dir/%x_slurm_%j.out and #SBATCH --error=output_dir/%x_error_slurm_%j.out

display_help() {
  echo ""
  echo "SLURM batch script plotting profiles."
  echo ""
  echo "Usage: sbatch -J job_name --output slurm_output_logs --error slurm_error_logs plot_profiles_sbatch_jsc.sh -c config.yml -p plot_dir -l log_dir [-j job_name]"
  echo ""
  echo " Options:"
  echo " -c    YAML configuration file for CASTLE network."
  echo " -p    Output directory for plots."
  echo " -l    Output directory for Python logs."
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
found_l=0
found_p=0
found_j=0

# Parse options
while getopts "c:l:p:j:h" opt; do
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
  p)
    found_p=1
    PLOT_DIR=$OPTARG
    ;;
  l)
    found_l=1
    PYTHON_DIR=$OPTARG
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
elif ((found_l == 0)); then
  echo -e "\nError: Failed to provide output directory for Python logs.\n"
  error_exit
elif ((found_p == 0)); then
  echo -e "\nError: Failed to provide output directory for plots.\n"
  error_exit
fi

if ((found_j == 0)); then
  JOB_NAME="plot_profiles"
fi

##################################
# Start computing shapley values #
##################################

PROJECT_ROOT="$(dirname "${PWD}")"

echo "Start time: "$(date)

conda run --cwd "$PROJECT_ROOT" --no-capture-output -n kuehbacher1_py3.9_tf python -u -m castle_offline_evaluation.main_castle_plot_profiles -c "$CONFIG" -p "$PLOT_DIR" >"${PYTHON_DIR}/${JOB_NAME}_python_${SLURM_JOB_ID}.out"
