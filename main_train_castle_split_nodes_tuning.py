import argparse
import datetime
import time
from pathlib import Path
import nni

import tensorflow as tf

from neural_networks.tuning.models_split_over_nodes_tuning import generate_models
from neural_networks.tuning.training_tuning import train_all_models
from neural_networks.tuning.training_mirrored_strategy_tuning import train_all_models as train_all_models_mirrored
from utils.setup import SetupNeuralNetworks


def train_castle(config_file, nn_inputs_file, nn_outputs_file, metric):
    argv = ["-c", config_file]
    setup = SetupNeuralNetworks(argv)

    inputs = _read_txt_to_list(nn_inputs_file)
    outputs = _read_txt_to_list(nn_outputs_file)

    params = {
        "num_hidden_layers": 4,
        "dense_units": 64,
        "activation_type": 'leakyrelu',
        "learning_rate": 1e-3,
        "learning_rate_schedule": {"schedule": "exp", "step": 5, "divide": 3},
        "lambda_weight": 1.0,
        "output_index": 64
    }

    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(f"Optimized parameters: {params}")

    selected_output = [outputs[params["output_index"]]]
    model_descriptions = generate_models(setup, inputs, selected_output, params)

    if setup.distribute_strategy == "mirrored" or setup.distribute_strategy == "multi_worker_mirrored":
        train_all_models_mirrored(model_descriptions, setup, tuning_params=params, tuning_metric=metric)
    else:
        train_all_models(model_descriptions, setup, tuning_params=params, tuning_metric=metric)


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

    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("-c", "--config_file", help="YAML configuration file for neural network creation.",
                               required=True)
    required_args.add_argument("-i", "--inputs_file", help=".txt file with NN inputs list.", required=True, type=str)
    required_args.add_argument("-o", "--outputs_file", help=".txt file with NN outputs list.", required=True, type=str)
    required_args.add_argument("-p", "--tuning_metric",
                               help="Metric used to measure tuning performance (e.g. 'val_loss', 'val_prediction_loss').",
                               required=True, type=str)

    args = parser.parse_args()

    yaml_config_file = Path(args.config_file)
    inputs_file = Path(args.inputs_file)
    outputs_file = Path(args.outputs_file)
    random_seed_parsed = args.seed
    tuning_metric = args.tuning_metric

    if not yaml_config_file.suffix == ".yml":
        parser.error(f"Configuration file must be YAML file (.yml). Got {yaml_config_file}")
    if not inputs_file.suffix == ".txt":
        parser.error(f"File with neural network inputs must be .txt file. Got {inputs_file}")
    if not outputs_file.suffix == ".txt":
        parser.error(f"File with neural network outputs must be .txt file. Got {outputs_file}")

    if random_seed_parsed is False:
        pass
    else:
        if random_seed_parsed is True:
            random_seed = 42
        else:
            random_seed = random_seed_parsed
        print(f"\n\nSet Tensorflow random seed for reproducibility: seed={random_seed}", flush=True)
        tf.random.set_seed(random_seed)

    print(f"\nYAML config file:      {yaml_config_file}")
    print(f"Input list .txt file:  {inputs_file}")
    print(f"Output list .txt file: {outputs_file}")
    print(f"Tuning metric:         {tuning_metric}")

    print(f"\n\n{datetime.datetime.now()} --- Start CASTLE training over multiple SLURM nodes.", flush=True)
    t_init = time.time()

    train_castle(yaml_config_file, inputs_file, outputs_file, tuning_metric)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
