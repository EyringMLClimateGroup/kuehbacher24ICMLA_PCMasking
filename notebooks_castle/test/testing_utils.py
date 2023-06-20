import os
import shutil

import tensorflow as tf


def delete_dir(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)


def set_memory_growth_gpu():
    physical_devices = tf.config.list_physical_devices("GPU")
    print(f"Number of GPUs: {len(physical_devices)}", flush=True)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
