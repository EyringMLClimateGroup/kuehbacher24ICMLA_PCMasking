import argparse
import datetime
import time
from pathlib import Path

import tensorflow as tf

from castle_offline_evaluation.castle_evaluation_utils import set_memory_growth_gpu, read_txt_to_list, read_txt_to_dict
from castle_offline_evaluation.castle_shapley_values import shap_single_variable, get_save_str, save_shapley_dict, \
    fill_shapley_dict
from utils.variable import Variable_Lev_Metadata


def compute_shapley(variables, config_file, var2index, n_time, n_samples, metric, save_dir):
    for var in variables:
        print(f"\n----")
        t_init_per_var = time.time()

        results = shap_single_variable(var, config_file, n_time, n_samples, metric)
        shap_dict = fill_shapley_dict(results, metric)
        save_shapley_dict(save_dir, Variable_Lev_Metadata.parse_var_name(var), shap_dict, var2index)

        t_end_per_var = time.time() - t_init_per_var
        print(f"\nElapsed time: {datetime.timedelta(seconds=t_end_per_var)}\n----\n")
    return

if __name__ == "__main__":
    # Allow memory growth for GPUs (this seems to be very important, because errors occur otherwise)
    if len(tf.config.list_physical_devices("GPU")):
        print(f"\nAllow memory growth on GPUs.", flush=True)
        set_memory_growth_gpu()

    parser = argparse.ArgumentParser(description="Computes shapley values using SHAP package.")
    required_args = parser.add_argument_group("setup arguments")
    required_args.add_argument("-c", "--config_file", help="YAML configuration file for neural network creation.",
                               required=True)
    required_args.add_argument("-o", "--outputs_file",
                               help=".txt file specifying the output variables for which networks the "
                                    "shapley values are to be computed",
                               required=True, type=str)
    required_args.add_argument("-m", "--outputs_map_file",
                               help=".txt file specifying the mapping between variable names and saved network names.",
                               required=True, type=str)
    required_args.add_argument("-p", "--plot_directory",
                               help="Output directory for shapley dictionaries and plots",
                               required=True, type=str)

    shap_args = parser.add_argument_group("shapley computation arguments")
    shap_args.add_argument("-t", "--n_time",
                           help="Number of time samples to select from the data.",
                           required=True, type=int)
    shap_args.add_argument("-s", "--n_samples",
                           help="Number of samples to be used for shapley computation.",
                           required=True, type=int)
    shap_args.add_argument("-e", "--metric",
                           help="Metric to be used on shapley values. Can be one of ['mean', 'abs_mean', 'abs_mean_sign'].",
                           required=True, type=str)

    args = parser.parse_args()

    yaml_config_file = Path(args.config_file)
    outputs_file = Path(args.outputs_file)
    map_file = Path(args.outputs_map_file)
    plot_dir = Path(args.plot_directory)

    # i_time is always "range" for shapley
    n_time = args.n_time
    n_samples = args.n_samples
    metric = args.metric

    if not yaml_config_file.suffix == ".yml":
        parser.error(f"Configuration file must be YAML file (.yml). Got {yaml_config_file}")
    if not outputs_file.suffix == ".txt":
        parser.error(f"File with output variable networks must be .txt file. Got {outputs_file}")
    if not map_file.suffix == ".txt":
        parser.error(f"File with outputs mapping be .txt file. Got {map_file}")

    output_vars = read_txt_to_list(outputs_file)
    output_map_dict = read_txt_to_dict(map_file)

    save_dir = Path(plot_dir,
                    get_save_str(idx_time="range", num_time=n_time, num_samples=n_samples, shap_metric=metric))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n\nYAML config file:           {yaml_config_file}")
    print(f"Output networks:            {output_vars}")
    print(f"Save directory:             {save_dir}")
    print(f"Number of time steps:       {n_time}")
    print(f"Number of samples:          {n_samples}")
    print(f"Averaging metric:           {metric}\n")

    print(f"\n\n{datetime.datetime.now()} --- Start CASTLE shapley computation for {len(output_vars)} networks.",
          flush=True)
    t_init = time.time()

    compute_shapley(output_vars, yaml_config_file, output_map_dict, n_time, n_samples, metric, save_dir)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Total elapsed time: {t_total}")
