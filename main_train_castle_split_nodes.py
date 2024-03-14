import argparse
import datetime
import os
import pickle
import time
from pathlib import Path

import numpy as np

from neural_networks.models_split_over_nodes import generate_models
from neural_networks.training import train_all_models
from utils.setup import SetupNeuralNetworks
from utils.tf_gpu_management import manage_gpu, set_tf_random_seed, set_gpu


def train_castle(config_file, nn_inputs_file, nn_outputs_file, train_indices, load_weights_from_ckpt,
                 continue_previous_training, config_fine_tune, seed):
    argv = ["-c", config_file]
    setup = SetupNeuralNetworks(argv)

    inputs = _read_txt_to_list(nn_inputs_file)
    outputs = _read_txt_to_list(nn_outputs_file)

    selected_outputs = [outputs[i] for i in train_indices]
    model_descriptions = generate_models(setup, inputs, selected_outputs, continue_training=continue_previous_training,
                                         seed=seed)

    # If we are doing fine-tuning, we need to load the weights from trained PreMaskNet
    if setup.nn_type == "MaskNet" and config_fine_tune is not None:
        load_fine_tune_weights(config_fine_tune, inputs, selected_outputs, model_descriptions, seed)

    history = train_all_models(model_descriptions, setup, from_checkpoint=load_weights_from_ckpt,
                               continue_training=continue_previous_training)

    # Save histories
    base_dir = Path(config_file).parent
    _save_history(history, model_descriptions, selected_outputs, base_dir)

    # In case of PreMaskNet, save the masking vector for each variable
    if setup.nn_type == "PreMaskNet":
        _save_masking_vector(model_descriptions, selected_outputs, base_dir)


def load_fine_tune_weights(config_fine_tune, inputs, selected_outputs, model_descriptions, seed):
    argv = ["-c", config_fine_tune]
    pre_mask_net_setup = SetupNeuralNetworks(argv)

    pre_mask_net_md = generate_models(pre_mask_net_setup, inputs, selected_outputs, seed=seed)

    for idx, md in enumerate(pre_mask_net_md):
        pre_mask_net = md.model

        weights_path = md.get_path(pre_mask_net_setup.nn_output_path)
        filename = md.get_filename() + "_weights.h5"

        print(f"\nLoading model weights from file {os.path.join(weights_path, filename)}")

        pre_mask_net.load_weights(os.path.join(weights_path, filename))

        for jdx, layer in enumerate(pre_mask_net.shared_hidden_layers):
            model_descriptions[idx].model.shared_hidden_layers[jdx].set_weights(layer.get_weights())

        model_descriptions[idx].model.output_layer.set_weights(pre_mask_net.output_layer.get_weights())

    del pre_mask_net_md
    del pre_mask_net_setup


def _save_masking_vector(model_descriptions, selected_outputs, base_dir):
    mv_dir = os.path.join(base_dir, "masking_vectors")
    Path(mv_dir).mkdir(parents=True, exist_ok=True)

    for i in range(len(model_descriptions)):
        var = selected_outputs[i]
        model = model_descriptions[i].model

        input_layer_weight = model.input_layer.trainable_variables[0].numpy()
        masking_vector = np.linalg.norm(input_layer_weight, ord=2, axis=1)

        fn = f"masking_vector_{var}.npy"
        np.save(os.path.join(mv_dir, fn), masking_vector)

        print(f'\nSaving masking vector {Path(*Path(os.path.join(mv_dir, fn)).parts[-3:])}.')


def _save_history(history, model_descriptions, selected_outputs, base_dir):
    for i in range(len(model_descriptions)):
        h = history[selected_outputs[i]].history

        # Save history
        f_name = model_descriptions[i].get_filename() + '_history.p'

        out_path = os.path.join(base_dir, "history")
        Path(out_path).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(out_path, f_name), 'wb') as out_f:
            pickle.dump(h, out_f)
        print(f"\n\nSaving history file {Path(*Path(os.path.join(out_path, f_name)).parts[-4:])}")


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
    parser.add_argument("-f", "--fine_tune_config",
                        help="Config for trained PreMaskNet to load weights for fine-tuning.",
                        required=False, default=None, type=str, nargs='?')

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
    fine_tune_cfg = None if args.fine_tune_config == "" else args.fine_tune_config

    if not yaml_config_file.suffix == ".yml":
        parser.error(f"Configuration file must be YAML file (.yml). Got {yaml_config_file}")
    if not inputs_file.suffix == ".txt":
        parser.error(f"File with neural network inputs must be .txt file. Got {inputs_file}")
    if not outputs_file.suffix == ".txt":
        parser.error(f"File with neural network outputs must be .txt file. Got {outputs_file}")

    if fine_tune_cfg is not None:
        if not fine_tune_cfg.endswith(".yml"):
            parser.error(f"Fine-tuning configuration file must be YAML file (.yml). Got {fine_tune_cfg}")
        else:
            fine_tune_cfg = Path(fine_tune_cfg)

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
    print(f"Train indices:         {train_idx}")
    print(f"Fine-tuning config:    {fine_tune_cfg}\n")

    print(f"\n\n{datetime.datetime.now()} --- Start CASTLE training over multiple SLURM nodes.", flush=True)
    t_init = time.time()

    train_castle(yaml_config_file, inputs_file, outputs_file, train_idx, load_ckpt, continue_training, fine_tune_cfg,
                 random_seed)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
