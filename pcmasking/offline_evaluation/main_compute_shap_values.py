import argparse
import datetime
import time
from pathlib import Path

import tensorflow as tf

from pcmasking.offline_evaluation.compute_shap_values import shap_single_variable, get_save_str, \
    save_shapley_dict, fill_shap_dict, save_shap_values
from pcmasking.offline_evaluation.evaluation_utils import set_memory_growth_gpu, parse_txt_to_list, \
    parse_txt_to_dict, parse_str_to_bool_or_int
from pcmasking.utils.variable import Variable_Lev_Metadata


def compute_shap(var, config_file, var2index, n_time, n_samples, metric, save_dir):
    """Computes the SHAP values for a specific variable and saves the results.

    Args:
        var (str): The output variable for which to compute SHAP values.
        config_file (str or Path): Path to the YAML configuration file for the neural network.
        var2index (dict): Dictionary mapping variable names to the indices used in output file names.
        n_time (int or str): Number of time samples to use for SHAP value computation.
        n_samples (int): Number of samples to be used for SHAP computation.
        metric (str): The metric to be used for SHAP values. One of ['mean', 'abs_mean', 'abs_mean_sign', 'none', 'all'].
        save_dir (str or Path): Directory where the SHAP value results will be saved.

    Returns:
        None: The function saves the computed Shapley values to the specified directory.
    """
    results = shap_single_variable(var, config_file, n_time, n_samples, metric)

    if metric == "none":
        save_shap_values(results, save_dir, Variable_Lev_Metadata.parse_var_name(variable), var2index)
    else:
        shap_dict = fill_shap_dict(results, metric)
        save_shapley_dict(save_dir, Variable_Lev_Metadata.parse_var_name(var), shap_dict, var2index)

    return


if __name__ == "__main__":
    """
    Main function to compute SHAP (SHapley Additive exPlanations) values for the trained neural network model 
    of one or more variables.
    
    Command-line Arguments:
       -x, --var_index (int, optional): Index of the output variable in outputs_file.txt for which to compute SHAP values.
                                        If not provided, SHAP values will be computed for all variables.
       -c, --config_file (str): Path to the YAML configuration file for neural network creation.
       -o, --outputs_file (str): Path to a text file specifying the output variable networks (.txt).
       -m, --outputs_map_file (str): Path to a text file specifying the mapping between variable names and saved network names.
       -p, --plot_directory (str): Path to the output directory where SHAP results and plots will be saved.
       -t, --n_time (int): Number of time samples to use for SHAP value computation.
       -s, --n_samples (int): Number of samples to use for SHAP value computation.
       -e, --metric (str): Metric to be used for SHAP values. One of ['mean', 'abs_mean', 'abs_mean_sign', 'none', 'all'].
    
    Variables:
       yaml_config_file (Path): Path object for the YAML configuration file.
       outputs_file (Path): Path object for the text file listing output variable networks.
       map_file (Path): Path object for the text file mapping variable names to saved network names.
       plot_dir (Path): Path object for the directory where SHAP results and plots will be saved.
       var_index (int or None): Index of the variable for SHAP computation. If None, computes for all variables.
       n_time (int): Number of time samples for SHAP computation.
       n_samples (int): Number of samples for SHAP computation.
       metric (str): The metric used for computing SHAP values.
       output_vars (list): List of output variables for SHAP computation.
       output_map_dict (dict): Dictionary mapping variable names to output indices.
       shap_for_variables (list): List of variables for which SHAP values will be computed.
       save_dir (Path): Directory where the computed SHAP values and plots will be saved.
    
    Raises:
       ValueError: If the variable index is not an integer or None.
       ArgumentError: If the configuration file or output files do not have the correct format.
    
    Example:
       To compute SHAP values for a single variable with index 0:
    
       $ python compute_shap.py --config_file config.yml --outputs_file outputs.txt --outputs_map_file map.txt \
           --plot_directory ./output --n_time 1440 --n_samples 1000 --metric mean --var_index 0
    
       To compute SHAP values for all variables:
    
       $ python main_compute_shap_values.py --config_file config.yml --outputs_file outputs.txt --outputs_map_file map.txt \
           --plot_directory ./output --n_time 1440 --n_samples 1000 --metric mean
    
    Workflow:
       1. Enable GPU memory growth (if GPUs are available) to avoid memory errors.
       2. Parse the command-line arguments to get the YAML configuration file, output variables, and SHAP computation settings.
       3. Check and ensure the provided file types are valid (e.g., YAML for configuration, TXT for output variables).
       4. Compute SHAP values for each variable in the specified list or for all variables if no specific index is given.
       5. Save the computed SHAP values and any related plots in the designated output directory.
       6. Print the elapsed time for SHAP computation for each variable and overall.
    """

    # Allow memory growth for GPUs (this seems to be very important, because errors occur otherwise)
    if len(tf.config.list_physical_devices("GPU")):
        print(f"\nAllow memory growth on GPUs.", flush=True)
        set_memory_growth_gpu()
    parser = argparse.ArgumentParser(description="Computes shapley values using SHAP package.")

    parser.add_argument("-x", "--var_index",
                        help="Index of the output variable in outputs_file.txt for which to compute the "
                             "Shapley values (int). If no index is given, SHAP values for all outputs "
                             "will be computed.",
                        required=False, type=int, nargs='?', default=None)

    required_args = parser.add_argument_group("setup arguments")
    required_args.add_argument("-c", "--config_file", help="YAML configuration file for neural network creation.",
                               required=True)
    required_args.add_argument("-o", "--outputs_file",
                               help="Text file specifying output variable networks (.txt).",
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
    output_map_dict = parse_txt_to_dict(map_file)

    if var_index is None:
        shap_for_variables = output_vars
    elif isinstance(var_index, int):
        shap_for_variables = [output_vars[var_index]]
    else:
        raise ValueError(f"Variable index must be integer or None. Got {var_index}")

    save_dir = Path(plot_dir,
                    get_save_str(idx_time="range", num_time=n_time, num_samples=n_samples, shap_metric=metric))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n\nYAML config file:        {yaml_config_file}")
    print(f"Networks:                {output_vars}")
    print(f"Save directory:          {save_dir}")
    print(f"Number of time steps:    {n_time}")
    print(f"Number of samples:       {n_samples}")
    print(f"Averaging metric:        {metric}\n")

    print(f"\n\n{datetime.datetime.now()} --- Start SHAP computation.", flush=True)
    t_init = time.time()

    for variable in shap_for_variables:
        print(f"\n--- Variable {variable}\n", flush=True)
        t_var_start = time.time()

        compute_shap(variable, yaml_config_file, output_map_dict, n_time, n_samples, metric, save_dir)

        t_var_total = datetime.timedelta(seconds=time.time() - t_init)
        print(f"\n--- Finished variable {variable}. Time elapsed {t_var_total}\n\n")

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Total elapsed time: {t_total}")
