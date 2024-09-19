import argparse
import datetime
import time
from pathlib import Path

import tensorflow as tf

from pcmasking.offline_evaluation.compute_stats import compute_stats
from pcmasking.offline_evaluation.evaluation_utils import set_memory_growth_gpu

if __name__ == "__main__":
    """
    Main method to compute statistics for a trained neural network model.

    This script reads the YAML configuration file, processes the command-line arguments to specify the 
    configuration and output directories, and computes various statistics for a neural network model 
    across specified time steps.

    Command-line Arguments:
        -c, --config_file (str): Path to the YAML configuration file for neural network creation.
        -p, --plot_directory (str): Path to the directory where the computed statistics files will be saved.

    Variables:
        i_time (str): Time index for statistics computation. Default is "range".
        n_time (int): Number of time steps for the statistics computation. Default is 1440.
        yaml_config_file (Path): Path to the configuration YAML file.
        plot_dir (Path): Path to the directory where the stats files will be saved.

    Raises:
        ArgumentError: If the provided configuration file is not a YAML file.

    Example:
        To compute statistics with a YAML configuration file and save the stats in the specified directory:

        $ python main_compute_stats.py --config_file config.yml --plot_directory ./output

    Workflow:
        1. Check and enable GPU memory growth for TensorFlow (if available).
        2. Parse command-line arguments for the YAML configuration file and output directory.
        3. Print the configuration details to the console.
        4. Call the `compute_stats` function to generate statistics for the neural network.
        5. Output the total elapsed time for the process.
    """
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
