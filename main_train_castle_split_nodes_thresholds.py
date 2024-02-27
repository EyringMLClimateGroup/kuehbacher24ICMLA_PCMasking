import argparse
import datetime
import gc
import time
from pathlib import Path

import numpy as np
import pickle
import os

from neural_networks.models_split_over_nodes import generate_models
from neural_networks.training import train_all_models
from utils.setup import SetupNeuralNetworks
from utils.tf_gpu_management import manage_gpu, set_tf_random_seed, set_gpu


def train_castle(config_file, nn_inputs_file, nn_outputs_file, train_index, percentile, load_weights_from_ckpt,
                 continue_previous_training, seed):
    base_dir = Path(config_file).parent

    argv = ["-c", config_file]
    setup = SetupNeuralNetworks(argv)

    inputs = _read_txt_to_list(nn_inputs_file)
    outputs = _read_txt_to_list(nn_outputs_file)

    selected_output = outputs[train_index]

    # Load masking vector and compute thresholds
    masking_vector_file = Path(str(setup.masking_vector_file).format(var=selected_output))
    print(f"\nLoading masking vector {(Path(*Path(masking_vector_file).parts[-4:]))}\n to get threshold bounds.")
    masking_vector = np.load(masking_vector_file)

    print(f"\nUsing thresholds between {1e-4} and {np.percentile(masking_vector, percentile)}.")
    thresholds = np.linspace(start=1e-4, stop=np.percentile(masking_vector, percentile), num=15, endpoint=False)

    # Run training for each threshold
    history_per_threshold = dict()

    for t in thresholds:
        print(f"\n\n\n--- Training with threshold {t}.\n\n")

        setup.mask_threshold = t

        model_descriptions = generate_models(setup, inputs, [selected_output],
                                             continue_training=continue_previous_training,
                                             seed=seed)

        h = train_all_models(model_descriptions, setup, from_checkpoint=load_weights_from_ckpt,
                             continue_training=continue_previous_training)
        history_per_threshold[t] = h[selected_output].history

        gc.collect()

    print(f"\n\n\n--- Finished training for all thresholds.")
    print("\nThreshold = {} for best last training loss = {}".format(
        *get_min_threshold_last_metric_element(history_per_threshold, "loss")))
    print("\nThreshold = {} for best validation loss = {}".format(
        *get_min_threshold_min_metric_element(history_per_threshold, "val_loss")))
    print("\n---")

    # Save history
    f_name = model_descriptions[0].get_filename() + '_history.p'
    out_path = os.path.join(base_dir, "threshold_histories")
    Path(out_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_path, f_name), 'wb') as out_f:
        pickle.dump(history_per_threshold, out_f)
    print(f"\n\nSaving history per threshold file {Path(*Path(os.path.join(out_path, f_name)).parts[-4:])}")


def get_min_threshold_last_metric_element(d, metric):
    min_key, min_history = min(d.items(), key=lambda x: x[1][metric][-1])
    return min_key, min_history[metric][-1]


def get_min_threshold_min_metric_element(d, metric):
    min_key, min_history = min(d.items(), key=lambda x: min(x[1][metric]))
    return min_key, min(min_history[metric])


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
    parser.add_argument("-g", "--gpu_index",
                        help="GPU index. If given, only the GPU specified by index will be used for training.",
                        required=False, default=False, type=int, nargs='?')

    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("-c", "--config_file", help="YAML configuration file for neural network creation.",
                               required=True)
    required_args.add_argument("-i", "--inputs_file", help=".txt file with NN inputs list.", required=True, type=str)
    required_args.add_argument("-o", "--outputs_file", help=".txt file with NN outputs list.", required=True, type=str)
    required_args.add_argument("-x", "--train_index", help="Index of network output variable in outputs list (int).",
                               required=True, type=int)
    required_args.add_argument("-r", "--percentile", help="Percentile of masking vector values used as upper bounds "
                                                          "for thresholds. Integer between 50-100.",
                               required=True, type=int)
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
    train_idx = args.train_index
    percentile = args.percentile
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

    # Check percentile value
    if not isinstance(percentile, int) or (not 50 <= percentile <= 100):
        raise ValueError("Given percentile was incorrect. Must be an integer value between 50-100.")

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
    print(f"Train index:           {train_idx}")
    print(f"Percentile:            {percentile}\n")

    print(f"\n\n{datetime.datetime.now()} --- Start model training.", flush=True)
    t_init = time.time()

    train_castle(yaml_config_file, inputs_file, outputs_file, train_idx, percentile, load_ckpt, continue_training,
                 random_seed)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")