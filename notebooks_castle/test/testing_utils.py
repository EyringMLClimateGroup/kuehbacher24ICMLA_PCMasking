import os
import shutil

import tensorflow as tf

from neural_networks.models import generate_models
from neural_networks.training_mirrored_strategy import train_all_models as train_all_models_mirrored
from neural_networks.training import train_all_models


def delete_dir(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)


def set_memory_growth_gpu():
    physical_devices = tf.config.list_physical_devices("GPU")
    print(f"Number of GPUs: {len(physical_devices)}", flush=True)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


def train_model_if_not_exists(setup):
    model_descriptions = generate_models(setup)

    for md in model_descriptions:
        trained_model = md.get_filename() + '_model.h5'
        training_path = str(md.get_path(setup.nn_output_path))
        if not os.path.isfile(os.path.join(training_path, trained_model)):
            if setup.do_mirrored_strategy:
                train_all_models_mirrored([md], setup)
            else:
                train_all_models([md], setup)
