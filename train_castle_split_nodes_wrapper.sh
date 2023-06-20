#!/bin/bash

# $1 should contain the .txt file with NN inputs
# $2 should contain the .txt file with NN outputs
# $3 should contain the number of nodes to be trained on
# $4 should contain the NN config file
if [ $# -eq 0 ]
  then
    echo "No arguments supplied, using default values specified in bash script."
    NN_INPUTS="..."
    NN_OUTPUTS="..."
    NUM_NODES=35
    NN_CONFIG="..."
  else
    NN_INPUTS=$1
    NN_OUTPUTS=$2
    NUM_NODES=$3
    NN_CONFIG=$4
fi

NUM_OUTPUTS="`grep -c ".*" $NN_OUTPUTS`"
NN_PER_NODE=$(($NUM_OUTPUTS/$NUM_NODES))

echo ""
echo "================================================================"
echo ""
echo "NN inputs file: $NN_INPUTS"
echo "NN outputs file: $NN_OUTPUTS"
echo "Number of training nodes: $NUM_NODES"
echo "NN config file: $NN_CONFIG"
echo "Number of NNs per node: $NN_PER_NODE"
echo ""
echo "================================================================"
echo ""

for (( i=0; i<$NUM_OUTPUTS; i+=$(($NN_PER_NODE+1)))); do
  END_INDEX=$(($i+$NN_PER_NODE))
  # Test if we've gone to far
  END_INDEX=$(( $(($NUM_OUTPUTS-1)) < $END_INDEX ? $(($NUM_OUTPUTS)) : $END_INDEX ))
  TRAIN_INDICES="$i-$END_INDEX"
  echo "Starting batch script with output indices $TRAIN_INDICES"

  sbatch train_castle_split_nodes_batch.sh "$NN_CONFIG" "$NN_INPUTS" "$NN_OUTPUTS" "$TRAIN_INDICES"
done