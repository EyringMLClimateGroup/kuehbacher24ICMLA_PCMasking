# Suppress tensorflow info logs
import datetime
import os
import time

if __name__ == "__main__":
    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not print
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from castle_offline_evaluation.castle_evaluation_utils import create_castle_model_description, parse_txt_to_dict

from pathlib import Path
import pickle

from neural_networks.load_models import load_single_model
from utils.setup import SetupDiagnostics
from utils.variable import Variable_Lev_Metadata


def plot_single_variable(var_name, i_time, n_time, lats, lons, stats, config, save_dir):
    argv = ["-c", config]
    setup = SetupDiagnostics(argv)

    var_model = load_single_model(setup, var_name)

    model_desc = create_castle_model_description(setup, var_model)

    var = Variable_Lev_Metadata.parse_var_name(var_name)
    if var.var.dimensions != 3:
        raise ValueError(f"Can only plot vertical cross-section for 3d variables. "
                         f"Given variable {var.var.value} has dimension {var.var.dimensions}.")
    var_keys = _get_var_keys(var.var.value, var_model.keys())

    model_desc.plot_double_profile(var, var_keys, itime=i_time, nTime=n_time,
                                   lats=lats, lons=lons, save=save_dir,
                                   stats=stats, show_plot=False)


def _get_var_keys(var_str, dict_keys):
    return [k for k in dict_keys if var_str in str(k)]


if __name__ == "__main__":
    ##########################################
    # Parameters
    i_time = "mean"
    n_time = 1
    lats = [-90, 90]
    lons = [0., 359.]
    stats = ["r2", "mse"]
    project_root = Path(__file__).parent.parent.resolve()

    config_file = Path(project_root,
                       "output_castle/training_28_custom_mirrored_functional/cfg_castle_training_run_2.yml")
    plot_dir = Path(project_root,
                    "output_castle/training_28_custom_mirrored_functional/plots_offline_evaluation/debug/plots_profiles/")

    variable = "tphystnd-691.39"
    ##########################################

    print(f"\n\n{datetime.datetime.now()} --- Start plotting profiles for variable {variable}.", flush=True)
    t_init = time.time()

    plot_single_variable(variable, i_time, n_time, lats, lons, stats, config_file, plot_dir)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
