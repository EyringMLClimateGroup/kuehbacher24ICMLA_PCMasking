import argparse
import datetime
import os
import time
from pathlib import Path

import tensorflow as tf

from castle_offline_evaluation.castle_evaluation_utils import set_memory_growth_gpu
from castle_offline_evaluation.castle_compute_stats import compute_stats

if __name__ == "__main__":
    # Allow memory growth for GPUs (this seems to be very important, because errors occur otherwise)
    if len(tf.config.list_physical_devices("GPU")):
        print(f"\nAllow memory growth on GPUs.", flush=True)
        set_memory_growth_gpu()

    parser = argparse.ArgumentParser(description="Computes stats.")
    required_args = parser.add_argument_group("setup arguments")
    required_args.add_argument("-c", "--config_file", help="YAML configuration file for neural network creation.",
                               required=True)
    required_args.add_argument("-p", "--plot_directory", help="Output directory for stats files",
                               required=True, type=str)

    args = parser.parse_args()

    yaml_config_file = Path(args.config_file)
    plot_dir = Path(args.plot_directory)

    i_time = "range"
    n_time = 1440

    if not yaml_config_file.suffix == ".yml":
        parser.error(f"Configuration file must be YAML file (.yml). Got {yaml_config_file}")

    print(f"\n\nYAML config file:       {yaml_config_file}")
    print(f"Plot directory:         {plot_dir}")
    print(f"Time index:             {i_time}")
    print(f"Number of time steps:   {n_time}\n\n")

    print(f"\n\n{datetime.datetime.now()} --- Start computing stats.", flush=True)
    t_init = time.time()

    compute_stats(i_time, n_time, yaml_config_file, plot_dir)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
