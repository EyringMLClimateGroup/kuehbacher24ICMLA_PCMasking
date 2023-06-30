#!/bin/bash
# Arguments
# $1 should contain the .txt file with NN inputs
# $2 should contain the .txt file with NN outputs
# $3 should contain the number of nodes to be trained on
# $4 should contain the NN config file

###########################
# Default argument values #
###########################
NN_INPUTS="output_castle/inputs_list.txt"
NN_OUTPUTS="output_castle/outputs_list.txt"
NUM_NODES=35
NN_CONFIG="nn_config/castle/test_cfg_castle_NN_Creation.yml"

#############
# Functions #
#############

display_help() {
  echo ""
  echo "Bash script wrapper for CASTLE training that splits training of model description list across multiple SLURM nodes."
  echo "Default options for arguments can be specified in this bash script."
  echo ""
  echo "Usage: $0 [-h] [-i inputs.txt] [-o outputs.txt] [-n nodes] [-c config.yml]"
  echo ""
  echo " Options:"
  echo " -i    txt file with input list for CASTLE networks."
  echo "       Current default: $NN_INPUTS"
  echo " -o    txt file with output list for CASTLE networks."
  echo "       Current default: $NN_OUTPUTS"
  echo " -n    Number of SLURM nodes to be used for training."
  echo "       Current default: $NUM_NODES"
  echo " -c    YAML configuration file for CASTLE networks."
  echo "       Current default: $NN_CONFIG"
  echo " -h    Print this help."
  echo
  exit 0
}

print_variables() {
  echo ""
  echo "================================================================="
  echo ""
  echo "  NN inputs file: $NN_INPUTS"
  echo "  NN outputs file: $NN_OUTPUTS"
  echo "  NN config file: $NN_CONFIG"
  echo "  Number of training nodes: $NUM_NODES"
  echo "  Number of NNs: $1"
  echo "  Number of NNs per node: $(($2 + 1))"
  echo ""
  echo "================================================================="
  echo ""
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

##########################
# No arguments are given #
##########################
if [ $# -eq 0 ]; then
  ###########################
  # Compute number of nodes #
  ###########################
  NUM_OUTPUTS="$(grep -c ".*" $NN_OUTPUTS)"
  NN_PER_NODE=$(($NUM_OUTPUTS / $NUM_NODES))

  echo -e "\nNo arguments were given. Default values are:"
  print_variables $NUM_OUTPUTS $NN_PER_NODE
  echo "Type 'y' if you want to use default values. Type 'n' to exit the script."
  echo -e "For more information use option -h.\n"

  read -r -e -p "Do you wish to proceed with defaults [y]/n: " input
  answer=${input:-"y"}

  echo ""

  if [ $answer == "n" ]; then
    graceful_exit
  fi

else
  ############################
  # Read and check arguments #
  ############################

  found_i=0
  found_o=0
  found_n=0
  found_c=0

  # Parse options
  while getopts "i:o:n:c:h" opt; do
    case ${opt} in
    h)
      display_help
      exit 0
      ;;
    i)
      found_i=1
      if [[ $OPTARG == *.txt ]]; then
        NN_INPUTS=$OPTARG
      else
        echo -e "\nError: Invalid value for option -i (NN inputs). Must be .txt file."
        error_exit
      fi
      ;;
    o)
      found_o=1
      if [[ $OPTARG == *.txt ]]; then
        NN_OUTPUTS=$OPTARG
      else
        echo -e "\nError: Invalid value for option -o (NN outputs). Must be .txt file."
        error_exit
      fi
      ;;
    n)
      found_n=1
      if (($OPTARG > 0)); then
        NUM_NODES=$OPTARG
      else
        echo -e "\nError: Invalid value for option -n (number of SLURM nodes). Must be greater than 0."
        error_exit
      fi
      ;;
    c)
      found_c=1
      if [[ $OPTARG == *.yml ]]; then
        NN_CONFIG=$OPTARG
      else
        echo -e "\nError: Invalid value for option -c (YAML config). Must be YAML file"
        error_exit
      fi
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

  ##########################################
  # Check if some arguments were not given #
  ##########################################
  echo ""

  # inputs txt
  if [[ $found_i == 0 ]]; then
    echo -e "\nNo input list supplied. Do you wish to use default value NN_INPUTS=$NN_INPUTS?"
    read -r -e -p "Enter [y]/n: " input
    answer=${input:-"y"}
    echo ""

    if [[ $answer == "n" ]]; then
      read -r -e -p "Please supply input list .txt file or press Enter to exit: " input
      if [[ $input == "" ]]; then
        graceful_exit
      else
        if [[ $input == *.txt ]]; then
          NN_INPUTS=$input
        else
          echo -e "\nError: Invalid value for NN inputs. Must be given in a .txt file."
          error_exit
        fi
      fi
    fi
  fi

  # outputs txt
  if [[ $found_o == 0 ]]; then
    echo -e "\nNo output list supplied. Do you wish to use default value NN_OUTPUTS=$NN_OUTPUTS?"
    read -r -e -p "Enter [y]/n: " input
    answer=${input:-"y"}
    echo ""

    if [[ $answer == "n" ]]; then
      read -r -e -p "Please supply output list .txt file or press Enter to exit: " input
      if [[ $input == "" ]]; then
        graceful_exit
      else
        if [[ $input == *.txt ]]; then
          NN_OUTPUTS=$input
        else
          echo -e "\nError: Invalid value for NN outputs. Must be given in a .txt file."
          error_exit
        fi
      fi
    fi
  fi

  # number of slurm nodes
  if [[ $found_n == 0 ]]; then
    echo -e "\nNumber of SLURM nodes not given. Do you wish to use default value NUM_NODES=$NUM_NODES?"
    read -r -e -p "Enter [y]/n: " input
    answer=${input:-"y"}
    echo ""

    if [[ $answer == "n" ]]; then
      read -r -e -p "Please type number of SLURM nodes or press Enter to exit: " input
      if [[ $input == "" ]]; then
        graceful_exit
      else
        if (($input > 0)); then
          NUM_NODES=$input
        else
          echo -e "\nError: Invalid value for number of SLURM nodes. Must be greater than 0."
          error_exit
        fi
      fi
    fi
  fi

  # yaml config file
  if [[ $found_c == 0 ]]; then
    echo -e "\nNo YAML config file given. Do you wish to use default value NN_CONFIG=$NN_CONFIG?"
    read -r -e -p "Enter [y]/n: " input
    answer=${input:-"y"}
    echo ""

    if [[ $answer == "n" ]]; then
      read -r -e -p "Please supply config .yml file or press Enter to exit: " input
      if [[ $input == "" ]]; then
        graceful_exit
      else
        if [[ $input == *.yml ]]; then
          NN_CONFIG=$input
        else
          echo "Error: Invalid value for YAML config. Must be YAML file"
          error_exit
        fi
      fi
    fi
  fi

  ###########################
  # Compute number of nodes #
  ###########################
  NUM_OUTPUTS="$(grep -c ".*" $NN_OUTPUTS)"
  NN_PER_NODE=$(($NUM_OUTPUTS / $NUM_NODES))

  echo -e "\n\nRunning script with the following variables:"
  print_variables $NUM_OUTPUTS $NN_PER_NODE
fi

#####################
# Start SLURM nodes #
#####################
for ((i = 0; i < $NUM_OUTPUTS; i += $(($NN_PER_NODE + 1)))); do
  END_INDEX=$(($i + $NN_PER_NODE))
  # Test if we've gone to far
  END_INDEX=$(($(($NUM_OUTPUTS - 1)) < $END_INDEX ? $(($NUM_OUTPUTS)) : $END_INDEX))
  TRAIN_INDICES="$i-$END_INDEX"
  echo "Starting batch script with output indices $TRAIN_INDICES"

  #sbatch train_castle_split_nodes_batch.sh "$NN_CONFIG" "$NN_INPUTS" "$NN_OUTPUTS" "$TRAIN_INDICES"
done

echo -e "\nFinished starting batch scripts.\n"
exit 0