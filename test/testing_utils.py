import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy import stats as stats

from neural_networks.data_generator import build_valid_generator
from neural_networks.models import generate_models
from neural_networks.training import train_all_models
from utils.variable import Variable_Lev_Metadata


def delete_output_dirs(model_description, setup):
    def _delete_output_dir(md):
        save_dir = md.get_path(setup.nn_output_path)
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir, ignore_errors=True)

    if isinstance(model_description, list):
        for el in model_description:
            _delete_output_dir(el)
    else:
        _delete_output_dir(model_description)

    tb_dir = setup.tensorboard_folder
    if os.path.isdir(tb_dir):
        shutil.rmtree(tb_dir, ignore_errors=True)


def set_strategy(setup, strategy):
    setup.distribute_strategy = strategy
    if strategy == "mirrored" and len(tf.config.list_physical_devices("GPU")) == 0:
        print("\nTensorflow found no physical devices. Cannot test distributed strategy without GPUs. "
              "Exiting test.")
        exit(0)
    return setup


def set_memory_growth_gpu():
    physical_devices = tf.config.list_physical_devices("GPU")
    print(f"Number of GPUs: {len(physical_devices)}", flush=True)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


def train_model_if_not_exists(setup):
    if setup.distribute_strategy == "mirrored" and not len(tf.config.list_physical_devices("GPU")):
        print("\nTensorflow found no physical devices. Cannot test distributed strategy without GPUs. "
              "Exiting test.")
        exit(0)
    model_descriptions = generate_models(setup)

    for md in model_descriptions:
        trained_model = md.get_filename() + '_model.keras'
        training_path = str(md.get_path(setup.nn_output_path))

        if not os.path.isfile(os.path.join(training_path, trained_model)):
            train_all_models([md], setup)


def build_test_gen(model_description, setup):
    input_vars_dict = model_description.input_vars_dict
    output_vars_dict = model_description.output_vars_dict
    if setup.distribute_strategy == "mirrored":
        num_replicas = model_description.strategy.num_replicas_in_sync
    else:
        num_replicas = None
    # Make sure that the batch size is small
    setup.use_val_batch_size = True
    setup.val_batch_size = 32

    return build_valid_generator(input_vars_dict, output_vars_dict, setup,
                                 input_pca_vars_dict=setup.input_pca_vars_dict,
                                 num_replicas_distributed=num_replicas)


def create_masking_vector(num_inputs, out_file, outputs_list):
    if not os.path.isdir((Path(out_file).parent)):
        Path(out_file.parent).mkdir(parents=True)

    np.random.seed(42)

    lower, upper = 0, 5
    mu, sigma = 0, 5

    masking_vector = stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma,
                                         size=(num_inputs,))

    np.save(out_file, masking_vector.astype(np.float32))

    for out_var in outputs_list:
        mv_file = Path(str(out_file).format(var=out_var))
        np.save(mv_file, masking_vector.astype(np.float32))

def create_threshold_file(out_file, outputs_list):
    if not os.path.isdir((Path(out_file).parent)):
        Path(out_file.parent).mkdir(parents=True)

    np.random.seed(42)

    lower, upper = 0, 3
    mu, sigma = 0, 3
    t = stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

    threshold_dict = dict()

    for out_var in outputs_list:
        threshold_dict[out_var] = t

    with open(out_file, "wb") as f:
        pickle.dump(threshold_dict, f)


def generate_output_var_list(setup):
    output_list = list()
    for spcam_var in setup.spcam_outputs:
        if spcam_var.dimensions == 3:
            for level, _ in setup.children_idx_levs:
                # There's enough info to build a Variable_Lev_Metadata list
                # However, it could be better to do a bigger reorganization
                var_name = f"{spcam_var.name}-{round(level, 2)}"
                output_list.append(Variable_Lev_Metadata.parse_var_name(var_name))
        elif spcam_var.dimensions == 2:
            var_name = spcam_var.name
            output_list.append(Variable_Lev_Metadata.parse_var_name(var_name))
    return output_list
