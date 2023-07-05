#!/bin/bash
# Arguments
# $1 should contain the .txt file with NN inputs
# $2 should contain the .txt file with NN outputs
# $3 should contain the number of nodes to be trained on
# $4 should contain the NN config file

###########################
# Default argument values #
###########################
NN_INPUTS="output_castle/training_3/inputs_list.txt"
NN_OUTPUTS="output_castle/training_3/outputs_list.txt"
NUM_NODES=20
NN_CONFIG="nn_config/castle/test_cfg_castle_NN_Creation.yml"
DISTRIBUTED=true

MAX_RUNNING_JOBS=20

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
  echo " -n    Number of SLURM nodes to be used for training. Maximum number of running jobs per user is $MAX_RUNNING_JOBS."
  echo "       Current default: $NUM_NODES"
  echo " -c    YAML configuration file for CASTLE networks."
  echo "       Current default: $NN_CONFIG"
  echo " -h    Print this help."
  echo ""
  exit 0
}

print_variables() {
  echo -e "\n================================================================="
  echo ""
  echo "  NN inputs file:                  $NN_INPUTS"
  echo "  NN outputs file:                 $NN_OUTPUTS"
  echo "  NN config file:                  $NN_CONFIG"
  echo "  Number of training nodes/jobs:   $NUM_NODES"
  echo ""
  echo -e "=================================================================\n\n"
}

print_computed_variables() {
  echo -e "\n================================================================="
  echo ""
  echo "  Number of NNs:               $NUM_OUTPUTS"
  echo "  Number of training nodes:    $NUM_NODES"
  echo "  Number of NNs per node/job:  $NN_PER_NODE"
  echo "  Using distributed training:  $DISTRIBUTED"
  echo ""
  echo -e "=================================================================\n\n"
}

case_counter() {
  case $1 in
  0)
    echo -e "\nError: Unknown input. Try again (2 tries left).\n"
    ;;
  1)
    echo -e "\nError: Unknown input. Try again (1 try left).\n"
    ;;
  2)
    echo -e "\nError: Unknown input.\n"
    error_exit
    ;;
  esac
}

more_nodes_than_jobs() {
  if (($1 > 20)); then
    echo -e "\nInfo: Cannot run $1 nodes/jobs simultaneously because it exceeds the maximum number of running jobs per user ($MAX_RUNNING_JOBS)."
    echo "  The jobs exceeding the job limit will be scheduled and will start once running jobs have finished."
    counter=0
    while [ $counter -lt 3 ]; do
      read -r -e -p "Do you wish to continue? [y]/n: " input

      if [[ $input == "y" || $input == "" ]]; then
        NUM_NODES=$1
        break
      elif [[ $input == "n" ]]; then
        graceful_exit
      else
        case_counter $counter
      fi
      counter=$(($counter + 1))
    done
  fi

}

compute_nn_per_node() {
  NUM_OUTPUTS="$(grep -c ".*" $NN_OUTPUTS)"
  NN_PER_NODE=$((($NUM_OUTPUTS + $NUM_NODES - 1) / $NUM_NODES))

  # Test if we have to use different number of nodes than specified
  TEMP=$((($NUM_OUTPUTS + $NN_PER_NODE - 1) / $NN_PER_NODE))

  if [ $NUM_NODES -ne $TEMP ]; then
    echo -e "\nInfo: Could not evenly split $NUM_OUTPUTS networks across $NUM_NODES nodes. Using $TEMP nodes."
    counter=0
    while [ $counter -lt 3 ]; do
      read -r -e -p "Do you want to continue? [y]/n: " input
      answer=${input:-"y"}

      if [[ $answer == "y" ]]; then
        NUM_NODES=$TEMP
        break
      elif [[ $answer == "n" ]]; then
        graceful_exit
      else
        case_counter $counter
      fi
      counter=$(($counter + 1))
    done
  fi
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
  echo -e "\nNo arguments were given. Using default values.\nFor more information about arguments use option -h."
  echo -e "\n\nDefault values are:"
  print_variables

  read -r -e -p "Do you wish to proceed with defaults [y]/n: " input
  answer=${input:-"y"}

  echo ""

  if [ $answer == "n" ]; then
    graceful_exit
  fi

  ###########################
  # Compute number of nodes #
  ###########################
  compute_nn_per_node

  echo -e "\n\nRunning script with the following variables:"
  print_computed_variables

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
        more_nodes_than_jobs $OPTARG
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
    outer_counter=0
    while [ $outer_counter -lt 3 ]; do
      read -r -e -p "Enter [y]/n: " input
      answer=${input:-"y"}
      echo ""

      if [[ $answer == "y" ]]; then
        break
      elif [[ $answer == "n" ]]; then
        inner_counter=0
        while [ $inner_counter -lt 3 ]; do
          read -r -e -p "Please supply input list .txt file or press Enter to exit: " input
          if [[ $input == "" ]]; then
            graceful_exit
          else
            if [[ $input == *.txt ]]; then
              NN_INPUTS=$input
              break 2
            else
              case $inner_counter in
              0)
                echo -e "\nError: Invalid value for NN inputs. Must be given in a .txt file. Try again (2 tries left).\n"
                ;;
              1)
                echo -e "\nError: Invalid value for NN inputs. Must be given in a .txt file. Try again (1 try left).\n"
                ;;
              2)
                echo -e "\nError: Invalid value for NN inputs. Must be given in a .txt file.\n"
                error_exit
                ;;
              esac

            fi
          fi
          inner_counter=$(($inner_counter + 1))
        done
      else
        #Unknown input
        case_counter $outer_counter
      fi
      outer_counter=$(($outer_counter + 1))
    done
  fi

  # outputs txt
  if [[ $found_o == 0 ]]; then
    echo -e "\nNo output list supplied. Do you wish to use default value NN_OUTPUTS=$NN_OUTPUTS?"
    outer_counter=0
    while [ $outer_counter -lt 3 ]; do
      read -r -e -p "Enter [y]/n: " input
      answer=${input:-"y"}
      echo ""

      if [[ $answer == "y" ]]; then
        break
      elif [[ $answer == "n" ]]; then
        inner_counter=0
        while [ $inner_counter -lt 3 ]; do
          read -r -e -p "Please supply output list .txt file or press Enter to exit: " input
          if [[ $input == "" ]]; then
            graceful_exit
          else
            if [[ $input == *.txt ]]; then
              NN_OUTPUTS=$input
              break 2
            else
              case $inner_counter in
              0)
                echo -e "\nError: Invalid value for NN outputs. Must be given in a .txt file. Try again (2 tries left).\n"
                ;;
              1)
                echo -e "\nError: Invalid value for NN outputs. Must be given in a .txt file. Try again (1 try left).\n"
                ;;
              2)
                echo -e "\nError: Invalid value for NN outputs. Must be given in a .txt file.\n"
                error_exit
                ;;
              esac
            fi
          fi
          inner_counter=$(($inner_counter + 1))
        done
      else
        #Unknown input
        case_counter $outer_counter
      fi
      outer_counter=$(($outer_counter + 1))
    done
  fi

  # number of slurm nodes
  if [[ $found_n == 0 ]]; then
    echo -e "\nNumber of SLURM nodes not given. Do you wish to use default value NUM_NODES=$NUM_NODES?"

    outer_counter=0
    while [ $outer_counter -lt 3 ]; do
      read -r -e -p "Enter [y]/n: " input
      answer=${input:-"y"}
      echo ""

      if [[ $answer == "y" ]]; then
        break
      elif [[ $answer == "n" ]]; then
        inner_counter=0
        while [ $inner_counter -lt 3 ]; do
          read -r -e -p "Please type number of SLURM nodes or press Enter to exit: " input
          if [[ $input == "" ]]; then
            graceful_exit
          else
            if (($input > 0)); then
              more_nodes_than_jobs $input
              break 2
            else
              case $inner_counter in
              0)
                echo -e "\nError: Invalid value for number of SLURM nodes. Must be greater than 0. Try again (2 tries left).\n"
                ;;
              1)
                echo -e "\nError: Invalid value for number of SLURM nodes. Must be greater than 0. Try again (1 try left).\n"
                ;;
              2)
                echo -e "\nError: Invalid value for number of SLURM nodes. Must be greater than 0.\n"
                error_exit
                ;;
              esac
            fi
          fi
          inner_counter=$(($inner_counter + 1))
        done
      else
        #Unknown input
        case_counter $outer_counter
      fi
      outer_counter=$(($outer_counter + 1))
    done
  fi

  # yaml config file
  if [[ $found_c == 0 ]]; then
    echo -e "\nNo YAML config file given. Do you wish to use default value NN_CONFIG=$NN_CONFIG?"
    outer_counter=0
    while [ $outer_counter -lt 3 ]; do
      read -r -e -p "Enter [y]/n: " input
      answer=${input:-"y"}
      echo ""

      if [[ $answer == "y" ]]; then
        break

      elif [[ $answer == "n" ]]; then
        inner_counter=0
        while [ $inner_counter -lt 3 ]; do
          read -r -e -p "Please supply config .yml file or press Enter to exit: " input
          if [[ $input == "" ]]; then
            graceful_exit
          else
            if [[ $input == *.yml ]]; then
              NN_CONFIG=$input
              break 2
            else
              case $inner_counter in
              0)
                echo -e "\nError: Invalid value for YAML config. Must be YAML file. Try again (2 tries left).\n"
                ;;
              1)
                echo -e "\nError: Invalid value for YAML config. Must be YAML file. Try again (1 try left).\n"
                ;;
              2)
                echo -e "\nError: Invalid value for YAML config.\n"
                error_exit
                ;;
              esac
            fi
          fi

          inner_counter=$(($inner_counter + 1))
        done
      else
        #Unknown input
        case_counter $outer_counter
      fi
      outer_counter=$(($outer_counter + 1))
    done
  fi

  ###########################
  # Compute number of nodes #
  ###########################
  compute_nn_per_node

  echo -e "\n\nRunning script with the following variables:"
  print_computed_variables
fi

##########################
# Exceeding compute time #
##########################
if [ "$DISTRIBUTED" = true ]; then
  NETS_IN_TIME=3
else
  NETS_IN_TIME=2
fi

if (($NN_PER_NODE > $NETS_IN_TIME)); then
  echo -e "\nInfo: Training $NN_PER_NODE nets per node may lead to exceeding the node reservation time limit (12h for GPU partition)."
  echo "  From experience, we can train about 2 nets in 12 hours (3 nets with distributed training). "
  echo "  If the time limit is exceeded, training will be aborted and there may not be any log output."
  echo -e "  (Tensorboard and model weights files should still be saved.)\n"

  outer_counter=0
  while [ $outer_counter -lt 3 ]; do
    read -r -e -p "Do you wish to keep the current value of $NN_PER_NODE NNs per node or change it to $NETS_IN_TIME NNs per node? [change]/keep: " input

    if [[ $input == "change" || $input == "" ]]; then
      NN_PER_NODE=$NETS_IN_TIME
      NUM_NODES=$((($NUM_OUTPUTS + $NN_PER_NODE - 1) / $NN_PER_NODE))
      echo -e "\nContinuing with $NN_PER_NODE NNs per node and $NUM_NODES nodes.\n"

      if (($NUM_NODES > 20)); then
        echo -e "\nInfo: Cannot run $1 nodes/jobs simultaneously because it exceeds the maximum number of running jobs per user (max running jobs=$MAX_RUNNING_JOBS)."
        echo -e "  The jobs exceeding the job limit will be scheduled and will start once running jobs have finished.\n"

        inner_counter=0
        while [ $inner_counter -lt 3 ]; do
          read -r -e -p "Do you wish to continue? [y]/n: " input
          if [[ $input == "y" || $input == "" ]]; then
            break 2
          elif [[ $input == "n" ]]; then
            graceful_exit
          else
            case_counter $inner_counter
          fi
          inner_counter=$(($inner_counter + 1))
        done
      fi

    elif [[ $input == "keep" ]]; then
      echo -e "\nKeeping $NN_PER_NODE NNs per node and $NUM_NODES nodes.\n"
      break

    else
      case_counter $outer_counter
    fi
    outer_counter=$(($outer_counter + 1))
  done
fi

echo -e "\n\n================================================================="
echo ""
echo " Starting batch jobs..."
echo ""
echo -e "=================================================================\n"

#####################
# Start SLURM nodes #
#####################
for ((i = 0; i < $NUM_OUTPUTS; i += $NN_PER_NODE)); do
  END_INDEX=$(($i + $NN_PER_NODE - 1))
  # Test if we've gone too far
  END_INDEX=$(($((($NUM_OUTPUTS - 1) < $END_INDEX)) ? $(($NUM_OUTPUTS - 1)) : $END_INDEX))
  TRAIN_INDICES="$i-$END_INDEX"
  echo -e "Starting batch script with output indices $TRAIN_INDICES"

  sbatch train_castle_split_nodes_batch.sh "$NN_CONFIG" "$NN_INPUTS" "$NN_OUTPUTS" "$TRAIN_INDICES"
done

echo -e "\nFinished starting batch scripts.\n"
exit 0
