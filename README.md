# CASTLE Branch for Spurious Links Project


### Requirements

Requirements are listed in `dependencies.yml`.

### Parallel and Distributed Training

There are two options for parallelization of training CASTLE models:

- Splitting the training of a list of models **across multiple nodes**.  
  The training runs for the individual models are independent of each other. Therefore, this
  form of parallelization is simply done by splitting the list of models and running the
  training on multiple SLURM nodes. You can do this with `train_castle_split_nodes_wrapper.sh`

- Splitting the training of a single model **across multiple GPUs**.  
  This form of parallelization happens internally
  using [tf.distributed.MirroredStrategy](https://www.tensorflow.org/guide/distributed_training#mirroredstrategy)
  and is done when `do_mirrored_strategy=True` in the configuration file.

  > ⚠️ Currently, `do_mirrored_strategy` is only read for CASTLE models in `SetupNeuralNetworks`.

### Start Training Runs

There are three shell scripts, which can be used for training:

- `train_castle_batch.sh`
- `train_castle_split_nodes_wrapper.sh`
- `train_castle_split_nodes_batch.sh`

`train_castle_batch.sh` is used for training all networks on one SLURM nodes, while the latter two
can be used to parallelize training across multiple SLURM nodes.
All three scripts support distributed training
via [tf.distributed.MirroredStrategy](https://www.tensorflow.org/guide/distributed_training#mirroredstrategy)
if `do_mirrored_strategy=True` in the given configuration file.

> ℹ️ The following commands must be run from the root directory of the repository,
> i.e. `iglesias-suarez2yxx_spuriouslinks/`.
>

#### Simple Training

`train_castle_batch.sh` is a SLURM run script that calls `main_train_castle.py`.
The configuration file for the run has to be set in the main file and there are no other arguments required.
To start training, run

```shell
sbatch train_castle_batch.sh
```

#### Parallel Training

In order to split the training across multiple SLURM nodes, you can use the shell
script `train_castle_split_nodes_wrapper.sh`.
You can make it executable with

```shell
chmod +x train_castle_split_nodes_wrapper.sh
```

and then run

```shell
./train_castle_split_nodes_wrapper -h
```

for usage instructions.

`train_castle_split_nodes_wrapper.sh` calls the SLURM run script `train_castle_split_nodes_batch.sh`.
In the latter you have to specify the output files for SLURM and Python logging.

You can also call `train_castle_split_nodes_batch.sh` directly, but check the expected command line arguments
in the file. There is no help for this script and the arguments are not checked for correctness.

In order to generate input/output lists in the form of .txt files automatically from the config file, run:

```shell
python -m main_generate_inputs_outputs_lists -i input_list.txt -o output_list.txt -c castle_config.yml -o outputs_mapping.txt
```

### Offline Evaluation

Jupyter notebooks with offline evaluation for CASTLE models can be found  `notebooks_castle_offline_evaluation/`.
Additionally, you can use Python scripts to run the same evaluations.
These scripts are located in `castle_offline_evaluation/`.
To run offline evaluation with Python scripts in the background, you can use

```shell
nohup python -m offline_eval_script > log_file.txt 2&>1 &
```

### Tests

You can find some tests for the CASTLE implementation in `notebooks_castle/test/`.
These are by no means complete and don't follow strict unit testing guidelines,
but they are useful for testing whether functionality has been destroyed after code changes.  

`notebooks_castle/test/config/` contains two configuration files that can be used for testing 
(they allow for quicker test runs, as the number of network inputs/outputs, hidden layers etc. 
is reduced).   

You can use the notebook `notebooks_castle/split_data.ipynb` to generate small datasets for 
testing from the normal training data. The path to the test data has to be specified in the 
config files. 

### Git Branch

To switch to this branch, run the command

```shell
git checkout castle
```

To pull or push on this branch, use

```shell
git pull origin castle
```

and

```shell
git push origin castle
```  

&nbsp;


> ℹ️ You can ommit `origin castle` if your branch pointer is currently on `castle`.
> You can check this with
> ```shell
> git log --oneline --decorate --graph --all
> ```
> or
> ```shell
> git status
> ```
> For more information on branches see
> this [Git branches guide](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell).


### SHAP package

Two lines in the SHAP package are changed:  
```diff
   112  assert type(self.model_output) != list, "The model output to be explained must be a single tensor!"
-  113  assert len(self.model_output.shape) < 3, "The model output must be a vector or a single value!"
+  113  assert len(self.model_output.shape) < 3 or (self.model_output.shape[-1] == 1 and len(
   114        self.model_output.shape) == 3), "The model output must be a vector or a single value!"
```

```diff
   744  op_handlers["Relu"] = nonlinearity_1d(0)
+  745  op_handlers["LeakyRelu"] = nonlinearity_1d(0)
```