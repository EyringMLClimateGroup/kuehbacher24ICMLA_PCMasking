import argparse
import datetime
import time
from pathlib import Path

import tensorflow as tf

from pcmasking.offline_evaluation.evaluation_utils import set_memory_growth_gpu
from pcmasking.offline_evaluation.plot_cross_section import plot_all_cross_sections

if __name__ == "__main__":
    """
    Main function to generate vertical cross-section plots for a trained neural network model.

    This script configures GPU memory settings, processes the command-line arguments to specify 
    the YAML configuration file and output directory, and generates cross-section plots for a neural 
    network based on various time steps, longitude indices, and metrics like MSE and R^2.

    Command-line Arguments:
        -c, --config_file (str): Path to the YAML configuration file for neural network creation.
        -p, --plot_directory (str): Path to the directory where the generated plots will be saved.

    Variables:
        i_time (str): The time index for the plots, default is "mean".
        n_time (int): Number of time steps to plot, default is 1440.
        i_lon (str): The longitude index, default is "mean".
        diff (bool): A flag indicating whether to compute difference plots, default is True.
        stats (list): List of statistics to compute and display in the plots, default is ["mse", "r2"].

    Raises:
        ArgumentError: If the provided configuration file is not a YAML (.yml) file.

    Example:
        $ python main_plot_cross_section.py --config_file config.yml --plot_directory ./output
        
    Workflow:
        1. Ensure GPU memory growth is enabled.
        2. Parse command-line arguments for the YAML configuration file and output directory.
        3. Print configuration details.
        4. Call the function `plot_all_cross_sections` to generate the plots based on the provided configuration.
        5. Output the total elapsed time for the process.
    """
    # Allow memory growth for GPUs (this seems to be very important, because errors occur otherwise)
    if len(tf.config.list_physical_devices("GPU")):
        print(f"\nAllow memory growth on GPUs.", flush=True)
        set_memory_growth_gpu()

    parser = argparse.ArgumentParser(description="Plots cross sections.")
    required_args = parser.add_argument_group("setup arguments")
    required_args.add_argument("-c", "--config_file", help="YAML configuration file for neural network creation.",
                               required=True)
    required_args.add_argument("-p", "--plot_directory", help="Output directory for plots.",
                               required=True, type=str)
    args = parser.parse_args()

    yaml_config_file = Path(args.config_file)
    plot_dir = Path(args.plot_directory)

    i_time = "mean"
    n_time = 1440
    i_lon = "mean"
    diff = True  # this doesn't matter
    stats = ["mse", "r2"]

    if not yaml_config_file.suffix == ".yml":
        parser.error(f"Configuration file must be YAML file (.yml). Got {yaml_config_file}")

    print(f"\n\nYAML config file:       {yaml_config_file}")
    print(f"Plot directory:         {plot_dir}")
    print(f"Time index:             {i_time}")
    print(f"Number of time steps:   {n_time}")
    print(f"Index longitude:        {i_lon}")
    print(f"Difference:             {diff}")
    print(f"Stats:                  {stats}\n")

    print(f"\n\n{datetime.datetime.now()} --- Start plotting cross sections.", flush=True)
    t_init = time.time()

    plot_all_cross_sections(i_time, n_time, i_lon, diff, stats, yaml_config_file, plot_dir)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
