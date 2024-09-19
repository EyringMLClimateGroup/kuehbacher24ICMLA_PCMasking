import argparse
import datetime
import time
from pathlib import Path

import tensorflow as tf

from pcmasking.offline_evaluation.evaluation_utils import set_memory_growth_gpu
from pcmasking.offline_evaluation.plot_lat_lon import plot_all_lat_lons

if __name__ == "__main__":
    """
    Main function to generate lat-lon plots for a trained neural network model.

    This script manages GPU memory, processes a YAML configuration file, and generates
    lat-lon plots, saving them to a specified directory. The process is customizable
    with options to calculate different statistics, handle multiple time steps, and compute
    plots with or without differences.

    Command-line Arguments:
        -c, --config_file (str): Path to the YAML configuration file for neural network creation.
        -p, --plot_directory (str): Path to the output directory for saving the generated plots.

    Variables:
        i_time (str): Specifies the time index to be used for plotting (default: "mean").
        diff (bool): Whether to compute difference plots (default: True).
        n_time (int): Number of time steps to include in the plot (default: 1440).
        stats (list): List of statistics to be computed and displayed in the plots (default: ["mse", "r2"]).

    Raises:
        ArgumentError: If the provided configuration file is not a YAML (.yml) file.

    Example:
        $ python main_plot_lat_lon.py --config_file config.yml --plot_directory ./output
        
    Workflow:
        1. Ensure GPU memory growth is enabled.
        2. Parse command-line arguments for the YAML configuration file and output directory.
        3. Print configuration details.
        4. Call the function `plot_all_lat_lons` to generate the plots based on the provided configuration.
        5. Output the total elapsed time for the process.   
    """
    # Allow memory growth for GPUs (this seems to be very important, because errors occur otherwise)
    if len(tf.config.list_physical_devices("GPU")):
        print(f"\nAllow memory growth on GPUs.", flush=True)
        set_memory_growth_gpu()

    parser = argparse.ArgumentParser(description="Plots lat-lon plots.")
    required_args = parser.add_argument_group("setup arguments")
    required_args.add_argument("-c", "--config_file", help="YAML configuration file for neural network creation.",
                               required=True)
    required_args.add_argument("-p", "--plot_directory", help="Output directory for plots.",
                               required=True, type=str)
    args = parser.parse_args()

    yaml_config_file = Path(args.config_file)
    plot_dir = Path(args.plot_directory)

    i_time = "mean"
    diff = True
    n_time = 1440
    stats = ["mse", "r2"]

    if not yaml_config_file.suffix == ".yml":
        parser.error(f"Configuration file must be YAML file (.yml). Got {yaml_config_file}")

    print(f"\n\nYAML config file:       {yaml_config_file}")
    print(f"Plot directory:         {plot_dir}")
    print(f"Time index:             {i_time}")
    print(f"Number of time steps:   {n_time}")
    print(f"Difference:             {diff}")
    print(f"Stats:                  {stats}\n")

    print(f"\n\n{datetime.datetime.now()} --- Start plotting lat-lon plots.", flush=True)
    t_init = time.time()

    plot_all_lat_lons(yaml_config_file, i_time=i_time, n_time=n_time, diff=diff, stats=stats, save_dir=plot_dir)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
