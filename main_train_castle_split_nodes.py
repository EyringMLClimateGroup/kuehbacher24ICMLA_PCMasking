import argparse
import datetime
import time
from pathlib import Path

import tensorflow as tf

from neural_networks.models_split_over_nodes import generate_models
from neural_networks.training import train_all_models
from neural_networks.training_mirrored_strategy import train_all_models as train_all_models_mirrored
from utils.setup import SetupNeuralNetworks


def train_castle(config_file, nn_inputs_file, nn_outputs_file, train_indices):
    argv = ["-c", config_file]
    setup = SetupNeuralNetworks(argv)

    inputs = _read_txt_to_list(nn_inputs_file)
    outputs = _read_txt_to_list(nn_outputs_file)

    selected_outputs = [outputs[i] for i in train_indices]
    model_descriptions = generate_models(setup, inputs, selected_outputs)

    if setup.distribute_strategy == "mirrored" or setup.distribute_strategy == "multi_worker_mirrored":
        train_all_models_mirrored(model_descriptions, setup)
    else:
        train_all_models(model_descriptions, setup)


def _read_txt_to_list(txt_file):
    line_list = list()
    with open(txt_file, 'r') as f:
        for line in f:
            line_list.append(line.rstrip())
    return line_list


def set_memory_growth_gpu():
    physical_devices = tf.config.list_physical_devices("GPU")
    print(f"\nNumber of GPUs: {len(physical_devices)}", flush=True)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


def parse_str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        # If it's not a boolean, we just pass it along and test for int later
        return v


if __name__ == "__main__":
    # Allow memory growth for GPUs (this seems to be very important, because errors occur otherwise)
    if len(tf.config.list_physical_devices("GPU")):
        print(f"\nAllow memory growth on GPUs.", flush=True)
        set_memory_growth_gpu()

    parser = argparse.ArgumentParser(description="Generates .txt files for neural network input and output "
                                                 "variables for specific setup configuration.")
    parser.add_argument("-s", "--seed", help="Integer value for random seed. "
                                             "Use 'False' or leave out this option to not set a random seed.",
                        default=False, type=parse_str_to_bool, nargs='?', const=True)

    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("-c", "--config_file", help="YAML configuration file for neural network creation.",
                               required=True)
    required_args.add_argument("-i", "--inputs_file", help=".txt file with NN inputs list.", required=True)
    required_args.add_argument("-o", "--outputs_file", help=".txt file with NN outputs list.", required=True)
    required_args.add_argument("-x", "--train_indices", help="Start and end index of outputs in outputs list, "
                                                             "specifying the neural networks that are to be trained. "
                                                             "Must be a string in the form 'start-end'.", required=True)

    args = parser.parse_args()

    yaml_config_file = Path(args.config_file)
    inputs_file = Path(args.inputs_file)
    outputs_file = Path(args.outputs_file)
    train_idx = args.train_indices
    random_seed_str = args.seed

    if not yaml_config_file.suffix == ".yml":
        parser.error(f"Configuration file must be YAML file (.yml). Got {yaml_config_file}")
    if not inputs_file.suffix == ".txt":
        parser.error(f"File with neural network inputs must be .txt file. Got {inputs_file}")
    if not outputs_file.suffix == ".txt":
        parser.error(f"File with neural network outputs must be .txt file. Got {outputs_file}")

    start, end = train_idx.split("-")
    train_idx = list(range(int(start), int(end) + 1))
    if not train_idx:
        raise ValueError("Given train indices were incorrect. Start indices must be smaller than end index. ")

    if random_seed_str is False:
        pass
    else:
        if random_seed_str is True:
            random_seed = 42

        else:
            try:
                random_seed = int(random_seed_str)
            except ValueError:
                raise ValueError(f"Invalid value given for random seed. Must be an integer, got {random_seed_str}. "
                                 f"Use 'False' or leave out option '-s' to not set a random seed. ")

        print(f"\n\nSet Tensorflow random seed for reproducibility: seed={random_seed}", flush=True)
        tf.random.set_seed(random_seed)

    print(f"\nYAML config file:      {yaml_config_file}")
    print(f"Input list .txt file:  {inputs_file}")
    print(f"Output list .txt file: {outputs_file}")
    print(f"Train indices:         {train_idx}\n")

    print(f"\n\n{datetime.datetime.now()} --- Start CASTLE training over multiple SLURM nodes.", flush=True)
    t_init = time.time()

    train_castle(yaml_config_file, inputs_file, outputs_file, train_idx)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
