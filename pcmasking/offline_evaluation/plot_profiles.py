import datetime
import gc
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt

from pcmasking.neural_networks.load_models import load_models
from pcmasking.neural_networks.model_diagnostics import ModelDiagnostics
from pcmasking.utils.setup import SetupDiagnostics
from pcmasking.utils.variable import Variable_Lev_Metadata


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
        models = models[model_key]

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
    # This main method is mainly used for testing

    ##########################################
    # Parameters
    i_time = "range"
    n_time = 1440
    lats = [-90, 90]
    lons = [0., 359.]
    stats = ["r2", "mse"]

    project_root = Path(__file__).parent.parent.parent.resolve()

    training_dir = Path("models/mask_net")
    config_file = os.path.join(project_root, training_dir, "cfg_mask_net_thresholds_train.yml")
    plot_dir = os.path.join(project_root, "plots_offline_evaluation/debug/profiles/")
    ##########################################

    print(f"\n\n{datetime.datetime.now()} --- Start plotting profiles.", flush=True)
    t_init = time.time()

    plot_profiles(i_time, n_time, lats, lons, stats, config_file, plot_dir)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
