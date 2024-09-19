import argparse
import datetime
import time
from pathlib import Path

import tensorflow as tf

from pcmasking.offline_evaluation.evaluation_utils import set_memory_growth_gpu
from pcmasking.offline_evaluation.plot_profiles import plot_profiles

if __name__ == "__main__":
    """
    Main function to plot horizontally averaged vertical profiles for a trained neural network. 

   Command-line Arguments:
       -c, --config_file (str): Path to the YAML configuration file for neural network creation.
       -p, --plot_directory (str): Path to the output directory for saving the generated profile plots.

   Variables:
       yaml_config_file (Path): Path object for the YAML configuration file.
       plot_dir (Path): Path object for the directory where profile plots will be saved.
       i_time (str): Time index for the profiles (default: "range").
       n_time (int): Number of time steps to use for plotting (default: 1440).
       lats (list of float): List of latitude bounds for the plots (default: [-90, 90]).
       lons (list of float): List of longitude bounds for the plots (default: [0., 359.]).
       stats (list of str): Statistical metrics to be used for plotting (default: ["r2", "mse"]).

   Raises:
       ArgumentError: If the configuration file does not have a .yml extension.

   Example:
       $ python main_plot_profiles.py --config_file config.yml --plot_directory ./output

   Workflow:
       1. Enable memory growth for GPUs if available to avoid memory allocation issues.
       2. Parse command-line arguments to get the YAML configuration file and the directory for output plots.
       3. Validate that the provided configuration file is in YAML format.
       4. Set up default parameters for the time index, number of time steps, latitude and longitude bounds, and metrics.
       5. Call the `plot_profiles` function to generate plots based on the configuration and save them to the specified directory.
       6. Print the total elapsed time for generating and saving the profile plots.
   """
    # Allow memory growth for GPUs (this seems to be very important, because errors occur otherwise)
    if len(tf.config.list_physical_devices("GPU")):
        print(f"\nAllow memory growth on GPUs.", flush=True)
        set_memory_growth_gpu()

    parser = argparse.ArgumentParser(description="Plots profiles.")
    required_args = parser.add_argument_group("setup arguments")
    required_args.add_argument("-c", "--config_file", help="YAML configuration file for neural network creation.",
                               required=True)
    required_args.add_argument("-p", "--plot_directory", help="Output directory for plots.",
                               required=True, type=str)

    args = parser.parse_args()

    yaml_config_file = Path(args.config_file)
    plot_dir = Path(args.plot_directory)

    i_time = "range"
    n_time = 1440
    lats = [-90, 90]
    lons = [0., 359.]
    stats = ["r2", "mse"]

    if not yaml_config_file.suffix == ".yml":
        parser.error(f"Configuration file must be YAML file (.yml). Got {yaml_config_file}")

    print(f"\n\nYAML config file:       {yaml_config_file}")
    print(f"Plot directory:         {plot_dir}")
    print(f"Time index:             {i_time}")
    print(f"Number of time steps:   {n_time}")
    print(f"Latitudes:              {lats}")
    print(f"Longitudes:             {lons}")
    print(f"Stats:                  {stats}\n")

    print(f"\n\n{datetime.datetime.now()} --- Start plotting profiles.", flush=True)
    t_init = time.time()

    plot_profiles(i_time, n_time, lats, lons, stats, yaml_config_file, plot_dir)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
