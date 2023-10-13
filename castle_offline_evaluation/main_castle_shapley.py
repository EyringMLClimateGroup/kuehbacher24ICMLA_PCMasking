import argparse
import datetime
import time
from pathlib import Path

import tensorflow as tf

from castle_offline_evaluation.castle_evaluation_utils import set_memory_growth_gpu, parse_txt_to_list, \
    parse_txt_to_dict, parse_str_to_bool_or_int
from castle_offline_evaluation.castle_shapley_values import shap_single_variable, get_save_str, save_shapley_dict, \
    fill_shapley_dict
from utils.variable import Variable_Lev_Metadata


def compute_shapley(var, config_file, var2index, n_time, n_samples, metric, save_dir):
    results = shap_single_variable(var, config_file, n_time, n_samples, metric)
    shap_dict = fill_shapley_dict(results, metric)
    save_shapley_dict(save_dir, Variable_Lev_Metadata.parse_var_name(var), shap_dict, var2index)

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
                               help="Text file specifying output variable networks (.txt).",
                               required=True, type=str)
    required_args.add_argument("-x", "--var_index",
                               help="Index of the output variable in outputs_file.txt for which to "
                                    "compute the Shapley values (int).",
                               required=True, type=int)
    required_args.add_argument("-m", "--outputs_map_file",
                               help=".txt file specifying the mapping between variable names and saved network names.",
                               required=True, type=str)
    required_args.add_argument("-p", "--plot_directory",
                               help="Output directory for shapley dictionaries and plots",
                               required=True, type=str)

    shap_args = parser.add_argument_group("shapley computation arguments")
    shap_args.add_argument("-t", "--n_time",
                           help="Number of time samples to select from the data.",
                           required=True, type=parse_str_to_bool_or_int)
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

    var_index = args.var_index

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

    output_vars = parse_txt_to_list(outputs_file)
    variable = output_vars[var_index]
    output_map_dict = parse_txt_to_dict(map_file)

    save_dir = Path(plot_dir,
                    get_save_str(idx_time="range", num_time=n_time, num_samples=n_samples, shap_metric=metric))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n\nYAML config file:           {yaml_config_file}")
    print(f"Output variable network:    {variable}")
    print(f"Save directory:             {save_dir}")
    print(f"Number of time steps:       {n_time}")
    print(f"Number of samples:          {n_samples}")
    print(f"Averaging metric:           {metric}\n")

    print(f"\n\n{datetime.datetime.now()} --- Start CASTLE shapley computation for variable {variable}.", flush=True)
    t_init = time.time()

    compute_shapley(variable, yaml_config_file, output_map_dict, n_time, n_samples, metric, save_dir)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Total elapsed time: {t_total}")
