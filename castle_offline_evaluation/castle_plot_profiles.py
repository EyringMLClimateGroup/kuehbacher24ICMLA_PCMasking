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

from pathlib import Path

from neural_networks.load_models import load_models
from neural_networks.model_diagnostics import ModelDiagnostics
from utils.setup import SetupDiagnostics
from utils.variable import Variable_Lev_Metadata
import matplotlib.pyplot as plt
import gc


def plot_profiles(i_time, n_time, lats, lons, stats, config, plot_dir):
    argv = ["-c", config]
    setup = SetupDiagnostics(argv)

    print("\nLoading models ...\n")
    models = load_models(setup, skip_causal_phq=True)
    model_key = setup.nn_type

    if setup.nn_type == "CausalSingleNN":
        dict_keys = models[model_key][setup.pc_alphas[0]][setup.thresholds[0]].keys()
        models = models[model_key][setup.pc_alphas[0]][setup.thresholds[0]]
    else:
        dict_keys = models[model_key].keys()
        models =  models[model_key]

    md = ModelDiagnostics(setup=setup, models=models)

    var_unit_str_three_d = [("tphystnd-3.64", "K/s"), ("phq-14.35", "kg/(kg*s)")]  # "phq-14.35", "phq-3.64"
    three_d_keys = [(Variable_Lev_Metadata.parse_var_name(var_str), unit) for var_str, unit in var_unit_str_three_d]

    print("\nComputing profile plots ...\n")
    for var, unit in three_d_keys:
        print(f"\n\n---- Variable {var}")
        var_keys = [k for k in dict_keys if var.var.value in str(k)]

        _ = md.plot_double_profile(var, var_keys, itime=i_time, nTime=n_time,
                                   lats=lats, lons=lons, save=plot_dir,
                                   stats=stats, show_plot=False, unit=unit)
        plt.close()
        gc.collect()


def _get_var_keys(var_str, dict_keys):
    return [k for k in dict_keys if var_str in str(k)]


if __name__ == "__main__":
    ##########################################
    # Parameters
    i_time = "range"
    n_time = 1440
    lats = [-90, 90]
    lons = [0., 359.]
    stats = ["r2", "mse"]

    project_root = Path(__file__).parent.parent.resolve()

    config_file = Path(project_root,
                       "output_castle/training_28_custom_mirrored_functional/cfg_castle_training_run_2.yml")
    plot_dir = Path(project_root,
                    "output_castle/training_28_custom_mirrored_functional/plots_offline_evaluation/debug/plots_profiles/")
    ##########################################

    print(f"\n\n{datetime.datetime.now()} --- Start plotting profiles.", flush=True)
    t_init = time.time()

    plot_profiles(i_time, n_time, lats, lons, stats, config_file, plot_dir)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
