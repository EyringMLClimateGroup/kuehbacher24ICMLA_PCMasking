# noinspection PyUnresolvedReferences
from utils.tf_gpu_management import set_memory_growth_gpu

import argparse
import datetime
import time
from pathlib import Path

import nni
import tensorflow as tf

from neural_networks.models_split_over_nodes import generate_models
from neural_networks.training import train_all_models
from utils.setup import SetupNeuralNetworks
from utils.variable import Variable_Lev_Metadata


def train_castle(config_file, nn_inputs_file, nn_outputs_file, var_idx, metric):
    argv = ["-c", config_file]
    setup = SetupNeuralNetworks(argv)

    inputs = _read_txt_to_list(nn_inputs_file)
    outputs = _read_txt_to_list(nn_outputs_file)

    params = {
        "hidden_layers": [64, 64, 64, 64],
        "activation": "leakyrelu",
        "learning_rate": 0.01,
        "learning_rate_schedule": {"schedule": "exp", "step": 5, "divide": 3},
        "kernel_initializer": "RandomNormal",
        "lambda_sparsity": 1.0,
    }

    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(f"\nOptimized parameters: {params}")

    # Set parameters in setup
    setup.hidden_layers = params["hidden_layers"]
    setup.activation = params["activation"]
    setup.init_lr = params["learning_rate"]
    setup.lr_schedule = params["learning_rate_schedule"]
    setup.lambda_sparsity = float(params["lambda_sparsity"])
    if params["kernel_initializer"] == "RandomNormal":
        setup.kernel_initializer_input_layers = {"initializer": "RandomNormal",
                                                 "mean": 0.0, "std": 0.01}
        setup.kernel_initializer_hidden_layers = {"initializer": "RandomNormal",
                                                  "mean": 0.0, "std": 0.1}
        setup.kernel_initializer_output_layers = {"initializer": "RandomNormal",
                                                  "mean": 0.0, "std": 0.01}
    elif params["kernel_initializer"] == "GlorotUniform":
        setup.kernel_initializer_input_layers = {"initializer": "GlorotUniform"}
        setup.kernel_initializer_hidden_layers = {"initializer": "GlorotUniform"}
        setup.kernel_initializer_output_layers = {"initializer": "GlorotUniform"}
    else:
        raise ValueError(f"Unknown value for kernel initializer: {params['kernel_initializer']}. "
                         f"Configured only for RandomNormal and GlorotUniform.")

    # Select output and generate model description
    selected_output = [outputs[var_idx]]
    print(f"\nTuning network for variable {selected_output[0]}.\n")

    model_descriptions = generate_models(setup, inputs, selected_output, params)

    # Train
    histories = train_all_models(model_descriptions, setup, tuning_params=params, tuning_metric=metric)

    final_metric = histories[Variable_Lev_Metadata.parse_var_name(selected_output)].history[metric][-1]
    nni.report_final_result(final_metric)
    print(f"\n\nFinal {metric} is {final_metric}\n")


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
    # Allow memory growth for GPUs (this seems to be very important, because errors occur otherwise)
    set_memory_growth_gpu()

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
    required_args.add_argument("-x", "--var_index",
                               help="Index of the output variable in outputs_file.txt for which to "
                                    "compute the Shapley values (int).",
                               required=True, type=int)
    required_args.add_argument("-p", "--tuning_metric",
                               help="Metric used to measure tuning performance (e.g. 'val_loss', 'val_prediction_loss').",
                               required=True, type=str)

    args = parser.parse_args()

    yaml_config_file = Path(args.config_file)
    inputs_file = Path(args.inputs_file)
    outputs_file = Path(args.outputs_file)
    var_index = args.var_index
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
    print(f"Output variable index: {var_index}")
    print(f"Tuning metric:         {tuning_metric}")

    print(f"\n\n{datetime.datetime.now()} --- Start CASTLE tuning.", flush=True)
    t_init = time.time()

    train_castle(yaml_config_file, inputs_file, outputs_file, var_index, tuning_metric)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
