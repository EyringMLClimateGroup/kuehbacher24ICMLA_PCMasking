import argparse
import datetime
import time
from pathlib import Path

from neural_networks.models_split_over_nodes import generate_models
from neural_networks.training import train_all_models
from utils.setup import SetupNeuralNetworks
from utils.tf_gpu_management import manage_gpu, set_tf_random_seed, set_gpu


def train_castle(config_file, nn_inputs_file, nn_outputs_file, train_indices, load_weights_from_ckpt,
                 continue_previous_training, seed):
    argv = ["-c", config_file]
    setup = SetupNeuralNetworks(argv)

    inputs = _read_txt_to_list(nn_inputs_file)
    outputs = _read_txt_to_list(nn_outputs_file)

    selected_outputs = [outputs[i] for i in train_indices]
    model_descriptions = generate_models(setup, inputs, selected_outputs, continue_training=continue_previous_training,
                                         seed=seed)

    train_all_models(model_descriptions, setup, from_checkpoint=load_weights_from_ckpt,
                     continue_training=continue_previous_training)


def _read_txt_to_list(txt_file):
    line_list = list()
    with open(txt_file, 'r') as f:
        for line in f:
            line_list.append(line.rstrip())
    return line_list


def parse_str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError(f"Could not parse {v} to boolean. See option -h for help.")


def parse_str_to_bool_or_int(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        try:
            return int(v)
        except ValueError:
            raise ValueError(f"Could not parse {v} to boolean or int.  See option -h for help.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates .txt files for neural network input and output "
                                                 "variables for specific setup configuration.")
    parser.add_argument("-s", "--seed", help="Integer value for random seed. "
                                             "Use 'False' or leave out this option to not set a random seed.",
                        default=False, type=parse_str_to_bool_or_int, nargs='?', const=True)
    parser.add_argument("-g", "--gpu_index", help="GPU index. If given, only the GPU specified by index will be used for training.",
                        required=False, default=False, type=int, nargs='?')

    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("-c", "--config_file", help="YAML configuration file for neural network creation.",
                               required=True)
    required_args.add_argument("-i", "--inputs_file", help=".txt file with NN inputs list.", required=True, type=str)
    required_args.add_argument("-o", "--outputs_file", help=".txt file with NN outputs list.", required=True, type=str)
    required_args.add_argument("-x", "--train_indices", help="Start and end index of outputs in outputs list, "
                                                             "specifying the neural networks that are to be trained. "
                                                             "Must be a string in the form 'start-end'.",
                               required=True, type=str)
    required_args.add_argument("-l", "--load_ckpt",
                               help="Boolean indicating whether to load weights from checkpoint from previous training.",
                               required=True, type=parse_str_to_bool)
    required_args.add_argument("-t", "--continue_training",
                               help="Boolean indicating whether to continue with previous training. The model "
                                    "(including optimizer) is loaded and the learning rate is initialized with the "
                                    "last learning rate from previous training.",
                               required=True, type=parse_str_to_bool)

    args = parser.parse_args()

    yaml_config_file = Path(args.config_file)
    inputs_file = Path(args.inputs_file)
    outputs_file = Path(args.outputs_file)
    train_idx = args.train_indices
    load_ckpt = args.load_ckpt
    continue_training = args.continue_training
    random_seed_parsed = args.seed
    gpu_index = args.gpu_index

    if not yaml_config_file.suffix == ".yml":
        parser.error(f"Configuration file must be YAML file (.yml). Got {yaml_config_file}")
    if not inputs_file.suffix == ".txt":
        parser.error(f"File with neural network inputs must be .txt file. Got {inputs_file}")
    if not outputs_file.suffix == ".txt":
        parser.error(f"File with neural network outputs must be .txt file. Got {outputs_file}")

    # GPU management: Allow memory growth if training is done on multiple GPUs, otherwise limit GPUs to single GPU
    if gpu_index is None:
        manage_gpu(yaml_config_file)
    else:
        set_gpu(index=gpu_index)

    # Parse indices of outputs selected for training
    start, end = train_idx.split("-")
    train_idx = list(range(int(start), int(end) + 1))
    if not train_idx:
        raise ValueError("Given train indices were incorrect. Start indices must be smaller than end index. ")

    # Set random seed
    if random_seed_parsed is False:
        random_seed = None
    else:
        if random_seed_parsed is True:
            random_seed = 42
        else:
            random_seed = random_seed_parsed
        set_tf_random_seed(random_seed)

    print(f"\nYAML config file:      {yaml_config_file}")
    print(f"Input list .txt file:  {inputs_file}")
    print(f"Output list .txt file: {outputs_file}")
    print(f"Train indices:         {train_idx}\n")

    print(f"\n\n{datetime.datetime.now()} --- Start CASTLE training over multiple SLURM nodes.", flush=True)
    t_init = time.time()

    train_castle(yaml_config_file, inputs_file, outputs_file, train_idx, load_ckpt, continue_training, random_seed)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
