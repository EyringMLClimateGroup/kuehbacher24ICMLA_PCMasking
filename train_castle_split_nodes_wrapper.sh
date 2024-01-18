#!/bin/bash

###########################
# Default argument values #
###########################
# todo: extract base dir from config?
base_dir="output_castle/training_42_gumbel_softmax_single_output_spars0.0001"
HPC="jsc" # jsc, dkrz

job_name_base="training_42_spars0.0001"

NN_INPUTS="${base_dir}/inputs_list.txt"
NN_OUTPUTS="${base_dir}/outputs_list.txt"
OUTPUTS_MAP="${base_dir}/outputs_map.txt"
NN_CONFIG="${base_dir}/cfg_gumbel_softmax_single_output.yml"

NUM_NODES=65
SEED=42
LOAD_CKPT="False"
CONTINUE_TRAINING="False"

MAX_RUNNING_JOBS_DKRZ=20

log_dir="${base_dir}/slurm_logs"
mkdir -p "$log_dir"

#############
# Functions #
#############

display_help() {
  echo ""
  echo "Bash script wrapper for CASTLE training that splits training of model description list across multiple SLURM nodes."
  echo "Training configuration parameters can either be specified in the script directly or via command line arguments."
  echo ""
  echo "Usage: $0 [-h] [-i inputs_list.txt] [-o outputs_list.txt] [-m outputs_map.txt] [-n nodes] [-c config.yml] [-l load_ckp_weight] [-t continue_training] [-p HPC] [-s seed]"
  echo ""
  echo " Options:"
  echo " -i    txt file with input list for CASTLE networks."
  echo "       Current value: $NN_INPUTS"
  echo ""
  echo " -o    txt file with output list for CASTLE networks."
  echo "       Current value: $NN_OUTPUTS"
  echo ""
  echo " -m    txt file with mapping for output variable identifiers for CASTLE networks."
  echo "       Current value: $OUTPUTS_MAP"
  echo ""
  echo " -n    Number of SLURM nodes to be used for training. For each node a SLURM job will be started."
  echo "       Current value: $NUM_NODES"
  echo "       Info: On DKRZ the maximum number of concurrently running jobs per user is 20."
  echo "             The jobs exceeding the job limit will be scheduled and will start once running jobs have finished."
  echo "             On JSC there is also a limit regarding the maximum number of running jobs per user."
  echo "             The precise values are adjusted to optimize system utilization."
  echo ""
  echo " -c    YAML configuration file for CASTLE networks."
  echo "       Current value: $NN_CONFIG"
  echo ""
  echo " -l    Boolean ('False' 'f', 'True', 't') indicating whether to load weights from checkpoint from previous training."
  echo "       Current value: $LOAD_CKPT"
  echo ""
  echo " -t    Boolean ('False' 'f', 'True', 't') indicating whether to continue with previous training. "
  echo "       The model (including optimizer) is loaded and the learning rate is initialized with the last learning rate from previous training."
  echo "       Current value: $CONTINUE_TRAINING"
  echo ""
  echo " -p    Which HPC one is working on ('dkrz' or 'jsc')."
  echo "       Current value: $HPC"
  echo ""
  echo " -s    Random seed. Leave out this option to not set a random seed."
  echo "       Current value: $SEED"
  echo ""
  echo " -h    Print this help."
  echo ""
  exit 0
}

print_variables() {
  echo -e "\n================================================================="
  echo ""
  echo "  NN inputs file:                 $NN_INPUTS"
  echo "  NN outputs file:                $NN_OUTPUTS"
  echo "  Outputs map file:               $OUTPUTS_MAP"
  echo "  NN config file:                 $NN_CONFIG"
  echo "  Number of training nodes/jobs:  $NUM_NODES"
  echo "  Load weights from checkpoint:   $LOAD_CKPT"
  echo "  Continue training:              $CONTINUE_TRAINING"
  echo "  Random seed:                    $SEED"
  echo ""
  echo -e "=================================================================\n\n"
}

print_computed_variables() {
  echo -e "\n================================================================="
  echo ""
  echo "  NN inputs file:                 $NN_INPUTS"
  echo "  NN outputs file:                $NN_OUTPUTS"
  echo "  Outputs map file:               $OUTPUTS_MAP"
  echo "  NN config file:                 $NN_CONFIG"
  echo "  Distributed training:           $DISTRIBUTED"
  echo "  Load weights from checkpoint:   $LOAD_CKPT"
  echo "  Continue training:              $CONTINUE_TRAINING"
  echo "  Random Seed:                    $SEED"
  echo ""
  echo "  Number of NNs:                  $NUM_OUTPUTS"
  echo "  Number of training nodes/jobs:  $NUM_NODES"
  echo "  Number of NNs per node/job:     $NN_PER_NODE"
  echo ""
  echo -e "=================================================================\n\n"
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

timestamp() {
  date +"%T" # current time
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

want_to_continue() {
  echo ""
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
}

compute_nn_per_node() {
  NUM_OUTPUTS="$(grep -c ".*" $NN_OUTPUTS)"
  # Check if it was empty
  if [[ $NUM_OUTPUTS == 0 ]]; then
    echo -e "\nError: Outputs .txt file was empty.\n"
    error_exit
  fi
  NN_PER_NODE=$((($NUM_OUTPUTS + $NUM_NODES - 1) / $NUM_NODES))

  # Test if we have to use different number of nodes than specified
  TEMP=$((($NUM_OUTPUTS + $NN_PER_NODE - 1) / $NN_PER_NODE))

  if [ $NUM_NODES -ne $TEMP ]; then
    echo -e "\nInfo: Could not evenly split $NUM_OUTPUTS networks across $NUM_NODES nodes. Using $TEMP nodes."
    want_to_continue
  fi
}

read_distributed() {
  if [ -f "$NN_CONFIG" ]; then
    TMP=$(grep 'distribute_strategy:' $NN_CONFIG)
    TMP=${TMP//*distribute_strategy: /}
    # Remove comments if there are any
    DISTRIBUTED=${TMP%#*}
    # Remove trailing new lines and spaces
    DISTRIBUTED="${DISTRIBUTED//[$'\t\r\n\" ']/}"
  else
    echo -e "\nError: YAML configuration file does not exist.\n"
    error_exit
  fi

  if [[ $DISTRIBUTED == "" ]]; then
    DISTRIBUTED="None"
  fi
}

check_inputs_file_exists() {
  if [ -f "$NN_INPUTS" ]; then
    :
  else
    echo -e "\nError: Inputs .txt file does not exist.\n"
    error_exit
  fi
}
check_map_file_exists() {
  if [ -f "$OUTPUTS_MAP" ]; then
    :
  else
    echo -e "\nError: Outputs map .txt file does not exist.\n"
    error_exit
  fi
}

check_outputs_file_exists() {
  # Check if outputs .txt file exists
  if [ -f "$NN_OUTPUTS" ]; then
    :
  else
    echo -e "\nError: Outputs .txt file does not exist.\n"
    error_exit
  fi
}

set_var_ident_str() {
  # Watch out: head starts at index 1
  start=$(($1 + 1))
  end=$(($2 + 1))

  var_ident_str=""
  concat=""

  # Also: i is used below, here we need to use j
  for ((j = start; j < end + 1; j += 1)); do
    var_line=$(head -n "$j" $NN_OUTPUTS | tail -1)
    var_line="${var_line//[$'\t\r\n ']/}"

    var_map=$(grep "$var_line" $OUTPUTS_MAP)
    var_ident=${var_map%:*}

    var_ident_str="${var_ident_str}${concat}${var_ident}"
    concat="-"
  done
  # set variable
  VAR_IDENT_STR=$var_ident_str
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

  #######################
  # Check files exists #
  #######################
  check_outputs_file_exists
  check_inputs_file_exists
  check_map_file_exists

  ###########################
  # Compute number of nodes #
  ###########################
  compute_nn_per_node

  #######################################
  # Read distributed strategy from YAML #
  #######################################
  read_distributed

  echo -e "\n\nRunning script with the following variables:"
  print_computed_variables

else
  ############################
  # Read and check arguments #
  ############################

  found_i=0
  found_o=0
  found_m=0
  found_n=0
  found_c=0
  found_s=0
  found_l=0
  found_t=0
  found_p=0

  # Parse options
  while getopts "i:o:m:n:c:l:t:p:s:h" opt; do
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
    m)
      found_m=1
      if [[ $OPTARG == *.txt ]]; then
        OUTPUTS_MAP=$OPTARG
      else
        echo -e "\nError: Invalid value for option -m (outputs map). Must be .txt file."
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
        echo -e "\nError: Invalid value for option -c (YAML config). Must be YAML file."
        error_exit
      fi
      ;;
    l)
      found_l=1
      lower_input=$(echo "$OPTARG" | tr '[:upper:]' '[:lower:]')
      if [[ $lower_input == "true" || $lower_input == "t" ]]; then
        LOAD_CKPT="True"
      elif [[ $lower_input == "false" || $lower_input == "f" ]]; then
        LOAD_CKPT="False"
      else
        echo -e "\nError: Invalid value for option -l (load from checkpoint). Must be a boolean ('True', 't', 'False', 'f')."
        error_exit
      fi
      ;;
    t)
      found_t=1
      lower_input=$(echo "$OPTARG" | tr '[:upper:]' '[:lower:]')
      if [[ $lower_input == "true" || $lower_input == "t" ]]; then
        CONTINUE_TRAINING="True"
      elif [[ $lower_input == "false" || $lower_input == "f" ]]; then
        CONTINUE_TRAINING="False"
      else
        echo -e "\nError: Invalid value for option -t (continue training). Must be a boolean ('True', 't', 'False', 'f')."
        error_exit
      fi
      ;;
    p)
      found_p=1
      lower_input=$(echo "$OPTARG" | tr '[:upper:]' '[:lower:]')
      if [[ $lower_input == "jsc" ]]; then
        HPC="jsc"
      elif [[ $lower_input == "dkrz" ]]; then
        HPC="dkrz"
      else
        echo -e "\nError: Invalid value for option -p (which HPC). Must either 'jsc' or 'dkrz'."
        error_exit
      fi
      ;;
    s)
      found_s=1
      re='^[+-]?[0-9]+$'
      if [[ $OPTARG =~ $re ]]; then
        SEED=$OPTARG
      else
        cap_input=$(echo "$OPTARG" | tr '[:lower:]' '[:upper:]')
        if [[ $cap_input == "NULL" ]]; then
          SEED="NULL"
        else
          echo -e "\nError: Invalid value for option -s (random seed). Must be an integer."
          error_exit
        fi
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

  # outputs map txt
  if [[ $found_m == 0 ]]; then
    echo -e "\nNo output map file supplied. Do you wish to use default value OUTPUTS_MAP=$OUTPUTS_MAP?"
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
          read -r -e -p "Please supply outputs map .txt file or press Enter to exit: " input
          if [[ $input == "" ]]; then
            graceful_exit
          else
            if [[ $input == *.txt ]]; then
              OUTPUTS_MAP=$input
              break 2
            else
              case $inner_counter in
              0)
                echo -e "\nError: Invalid value for option -m (outputs map). Must be .txt file. Try again (2 tries left).\n"
                ;;
              1)
                echo -e "\nError: Invalid value for option -m (outputs map). Must be .txt file. Try again (1 try left).\n"
                ;;
              2)
                echo -e "\nError: Invalid value for option -m (outputs map). Must be .txt file.\n"
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
              NUM_NODES=$input
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

  # load from checkpoint
  if [[ $found_l == 0 ]]; then
    echo -e "\nLoad from checkpoint not given. Do you wish to use default value LOAD_CKPT=$LOAD_CKPT?."

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
          read -r -e -p "Please type whether to load from checkpoint ('True', 't', 'False', 'f') or press Enter to exit: " input
          if [[ $input == "" ]]; then
            graceful_exit
          else
            lower_input=$(echo $input | tr '[:upper:]' '[:lower:]')
            if [[ $lower_input == "false" || $lower_input == "f" ]]; then
              LOAD_CKPT="False"
              break 2
            elif [[ $lower_input == "true" || $lower_input == "t" ]]; then
              LOAD_CKPT="True"
              break 2
            else
              case $inner_counter in
              0)
                echo -e "\nError: Invalid value for option -l (load from checkpoint). Must be a boolean ('True', 't', 'False', 'f'). Try again (2 tries left).\n"
                ;;
              1)
                echo -e "\nError: Invalid value for option -l (load from checkpoint). Must be a boolean ('True', 't', 'False', 'f'). Try again (1 try left).\n"
                ;;
              2)
                echo -e "\nError: Invalid value for option -l (load from checkpoint). Must be a boolean ('True', 't', 'False', 'f').\n"
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

  # continue training
  if [[ $found_t == 0 ]]; then
    echo -e "\nBoolean indication whether to continue training not given. Do you wish to use default value CONTINUE_TRAINING=$CONTINUE_TRAINING?."

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
          read -r -e -p "Please type whether to continue training ('True', 't', 'False', 'f') or press Enter to exit: " input
          if [[ $input == "" ]]; then
            graceful_exit
          else
            lower_input=$(echo $input | tr '[:upper:]' '[:lower:]')
            if [[ $lower_input == "false" || $lower_input == "f" ]]; then
              CONTINUE_TRAINING="False"
              break 2
            elif [[ $lower_input == "true" || $lower_input == "t" ]]; then
              CONTINUE_TRAINING="True"
              break 2
            else
              case $inner_counter in
              0)
                echo -e "\nError: Invalid value for option -t (continue training). Must be a boolean ('True', 't', 'False', 'f'). Try again (2 tries left).\n"
                ;;
              1)
                echo -e "\nError: Invalid value for option -t (continue training). Must be a boolean ('True', 't', 'False', 'f'). Try again (1 try left).\n"
                ;;
              2)
                echo -e "\nError: Invalid value for option -t (continue training). Must be a boolean ('True', 't', 'False', 'f').\n"
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

  # which HPC
  if [[ $found_p == 0 ]]; then
    echo -e "\nHPC not specified. Do you wish to use default value HPC=$HPC?."

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
          read -r -e -p "Please specify HPC ('jsc' or 'dkrz') or press Enter to exit: " input
          if [[ $input == "" ]]; then
            graceful_exit
          else
            lower_input=$(echo "$OPTARG" | tr '[:upper:]' '[:lower:]')
            if [[ $lower_input == "jsc" ]]; then
              HPC="jsc"
              break 2
            elif [[ $lower_input == "dkrz" ]]; then
              HPC="dkrz"
              break 2
            else
              case $inner_counter in
              0)
                echo -e "\nError: Invalid value for option -p (which HPC). Must either 'jsc' or 'dkrz'. Try again (2 tries left).\n"
                ;;
              1)
                echo -e "\nError: Invalid value for option -p (which HPC). Must either 'jsc' or 'dkrz'. Try again (1 try left).\n"
                ;;
              2)
                echo -e "\nError: Invalid value for option -p (which HPC). Must either 'jsc' or 'dkrz'.\n"
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

  # random seed
  if [[ $found_s == 0 ]]; then
    echo -e "\nRandom seed not given. Do you wish to use default value SEED=$SEED? If SEED is NULL, no random seed will be set."

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
          read -r -e -p "Please type a random seed or press Enter to exit: " input
          if [[ $input == "" ]]; then
            graceful_exit
          else
            re='^[+-]?[0-9]+$'
            if [[ $input =~ $re ]]; then
              SEED=$input
              break 2
            else
              cap_input=$(echo $input | tr '[:lower:]' '[:upper:]')
              if [[ $cap_input == "NULL" ]]; then
                SEED="NULL"
                break 2
              else
                case $inner_counter in
                0)
                  echo -e "\nError: Invalid value for option -s (random seed). Must be an integer or NULL. Try again (2 tries left).\n"
                  ;;
                1)
                  echo -e "\nError: Invalid value for option -s (random seed). Must be an integer or NULL. Try again (1 try left).\n"
                  ;;
                2)
                  echo -e "\nError: Invalid value for option -s (random seed). Must be an integer or NULL.\n"
                  error_exit
                  ;;
                esac
              fi
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

  #######################
  # Check files exists #
  #######################
  check_outputs_file_exists
  check_inputs_file_exists
  check_map_file_exists

  ###########################
  # Compute number of nodes #
  ###########################
  compute_nn_per_node

  #######################################
  # Read distributed strategy from YAML #
  #######################################
  read_distributed

  echo -e "\n\nRunning script with the following variables:"
  print_computed_variables
fi
###################################
# Check: Do you want to continue? #
###################################
want_to_continue

########################
# Random seed to false #
########################
if [[ $SEED == "NULL" ]]; then
  SEED="False"
fi

################
# Batch script #
################
if [[ $HPC == "jsc" ]]; then
  BATCH_SCRIPT="train_castle_split_nodes_batch_jsc.sh"
elif [[ $HPC == "dkrz" ]]; then
  BATCH_SCRIPT="train_castle_split_nodes_batch_dkrz.sh"
else
  echo -e "\nUnknown HPC ${HPC}."
  error_exit
fi

echo -e "\n\n$(timestamp) --- Starting SLURM jobs.\n"

#####################
# Start SLURM nodes #
#####################
for ((i = 0; i < $NUM_OUTPUTS; i += $NN_PER_NODE)); do
  END_INDEX=$(($i + $NN_PER_NODE - 1))
  # Test if we've gone too far
  END_INDEX=$(($((($NUM_OUTPUTS - 1) < $END_INDEX)) ? $(($NUM_OUTPUTS - 1)) : $END_INDEX))
  TRAIN_INDICES="$i-$END_INDEX"

  # Set variable VAR_IDENT_STR
  set_var_ident_str "$i" "$END_INDEX"
  JOB_NAME="${job_name_base}_${VAR_IDENT_STR}"
  # Check size of string (otherwise this may cause problems saving files
  if [[ ${#JOB_NAME} -gt 50 ]]; then
    JOB_NAME="training_model"
  fi

  slurm_o="${log_dir}/%x_slurm_%j.out"
  slurm_e="${log_dir}/%x_error_slurm_%j.out"

  echo -e "\nStarting batch script with output indices $TRAIN_INDICES"
  echo "Job name: ${JOB_NAME}"

  sbatch -J "$JOB_NAME" --output "$slurm_o" --error "$slurm_e" "$BATCH_SCRIPT" -c "$NN_CONFIG" -i "$NN_INPUTS" -o "$NN_OUTPUTS" -x "$TRAIN_INDICES" -l "$LOAD_CKPT" -t "$CONTINUE_TRAINING" -s "$SEED" -j "$JOB_NAME" -p "$log_dir"
done

echo -e "\n$(timestamp) --- Finished starting batch scripts.\n\n"
exit 0
