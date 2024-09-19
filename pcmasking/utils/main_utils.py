import os
import pickle
from pathlib import Path

import numpy as np

from pcmasking.neural_networks.models_split_over_nodes import generate_models as generate_models_subset
from pcmasking.neural_networks.models import  generate_models as generate_models_all
from pcmasking.utils.setup import SetupNeuralNetworks
from pcmasking.utils.tf_gpu_management import set_tf_random_seed


def load_fine_tune_weights(config_fine_tune, model_descriptions, seed, inputs=None, outputs=None):
    argv = ["-c", config_fine_tune]
    pre_mask_net_setup = SetupNeuralNetworks(argv)

    if inputs is None and outputs is None:
        pre_mask_net_md = generate_models_all(pre_mask_net_setup, seed=seed)
    else:
        pre_mask_net_md = generate_models_subset(pre_mask_net_setup, inputs, outputs, seed=seed)

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


def save_masking_vector(model_descriptions, selected_outputs, base_dir):
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


def save_history(history, model_descriptions, selected_outputs, base_dir):
    for i in range(len(model_descriptions)):
        h = history[selected_outputs[i]].history

        # Save history
        f_name = model_descriptions[i].get_filename() + '_history.p'

        out_path = os.path.join(base_dir, "history")
        Path(out_path).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(out_path, f_name), 'wb') as out_f:
            pickle.dump(h, out_f)
        print(f"\n\nSaving history file {Path(*Path(os.path.join(out_path, f_name)).parts[-4:])}")


def read_txt_to_list(txt_file):
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


def set_random_seed(random_seed_parsed):
    if random_seed_parsed is False:
        random_seed = None
    else:
        if random_seed_parsed is True:
            random_seed = 42
        else:
            random_seed = random_seed_parsed
        set_tf_random_seed(random_seed)
    return random_seed


def save_threshold_histories(base_dir, history_per_threshold, model_descriptions):
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
