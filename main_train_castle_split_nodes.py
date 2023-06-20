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

    if setup.do_mirrored_strategy:
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
    print(f"Number of GPUs: {len(physical_devices)}", flush=True)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


if __name__ == "__main__":
    # Allow memory growth for GPUs (this seems to be very important, because errors occur otherwise)
    if len(tf.config.list_physical_devices("GPU")):
        print(f"Allow memory growth on GPUs.", flush=True)
        set_memory_growth_gpu()

    parser = argparse.ArgumentParser(description="Generates .txt files for neural network input and output "
                                                 "variables for specific setup configuration.")
    parser.add_argument("-c", "--config_file", help="YAML configuration file for neural network creation.")
    parser.add_argument("-i", "--inputs_file", help=".txt file with NN inputs list.")
    parser.add_argument("-o", "--outputs_file", help=".txt file with NN outputs list.")
    parser.add_argument("-x", "--train_indices", help="Start and end index of outputs in outputs list, "
                                                      "specifying the neural networks that are to be trained. "
                                                      "Must be a string in the form 'start-end'.")

    args = parser.parse_args()

    yaml_config_file = Path(args.config_file)
    inputs_file = Path(args.inputs_file)
    outputs_file = Path(args.outputs_file)
    train_idx = args.train_indices

    print(f"{yaml_config_file=}")
    print(f"{inputs_file=}")
    print(f"{outputs_file=}")
    print(f"{train_idx=}")

    if not yaml_config_file.suffix == ".yml":
        parser.error(f"Configuration file must be YAML file (.yml). Got {yaml_config_file}")
    if not inputs_file.suffix == ".txt":
        parser.error(f"File with neural network inputs must be .txt file. Got {inputs_file}")
    if not outputs_file.suffix == ".txt":
        parser.error(f"File with neural network outputs must be .txt file. Got {outputs_file}")

    print(train_idx)
    start, end = train_idx.split("-")
    train_idx = list(range(int(start), int(end) + 1))
    if not train_idx:
        raise ValueError("Given train indices were incorrect. Start indices must be smaller than end index. ")

    print(f"Start CASTLE training over multiple SLURM nodes.", flush=True)
    t_init = time.time()

    print(train_idx)

    # train_castle(yaml_config_file, inputs_file, outputs_file, train_idx)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"{datetime.datetime.now()} Finished. Time: {t_total}")
