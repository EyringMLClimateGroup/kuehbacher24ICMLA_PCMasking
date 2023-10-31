import datetime
import time
from pathlib import Path

import tensorflow as tf

from neural_networks.models import generate_models
from neural_networks.training import train_all_models
from utils.setup import SetupNeuralNetworks


def train_castle():
    argv = ["-c", Path("nn_config", "castle", "cfg_castle_adapted.yml")]
    setup = SetupNeuralNetworks(argv)

    load_weights_from_ckpt = False
    continue_previous_training = False

    model_descriptions = generate_models(setup, continue_training=continue_previous_training)

    train_all_models(model_descriptions, setup, from_checkpoint=load_weights_from_ckpt,
                     continue_training=continue_previous_training)


def set_memory_growth_gpu():
    physical_devices = tf.config.list_physical_devices("GPU")
    print(f"Number of GPUs: {len(physical_devices)}", flush=True)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


if __name__ == "__main__":
    if len(tf.config.list_physical_devices("GPU")):
        print(f"Allow memory growth on GPUs.", flush=True)
        set_memory_growth_gpu()

    print(f"Start CASTLE training.", flush=True)
    t_init = time.time()
    train_castle()
    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"{datetime.datetime.now()} Finished. Time: {t_total}")
