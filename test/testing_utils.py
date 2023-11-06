import os
import shutil

import tensorflow as tf

from neural_networks.data_generator import build_valid_generator
from neural_networks.models import generate_models
from neural_networks.training import train_all_models


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
