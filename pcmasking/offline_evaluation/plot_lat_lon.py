import gc
import os
from pathlib import Path

import matplotlib.pyplot as plt

from pcmasking.neural_networks.load_models import load_single_model, load_models
from pcmasking.neural_networks.model_diagnostics import ModelDiagnostics
from pcmasking.offline_evaluation.evaluation_utils import create_model_description
from pcmasking.utils.setup import SetupDiagnostics
from pcmasking.utils.variable import Variable_Lev_Metadata


def plot_single_variable(var_name, config, i_time, n_time, diff, stats, vmin, vmax, save_dir):
    argv = ["-c", config]
    setup = SetupDiagnostics(argv)

    var_model = load_single_model(setup, var_name)

    model_desc = create_model_description(setup, var_model)

    var = Variable_Lev_Metadata.parse_var_name(var_name)
    if vmax and vmin:
        model_desc.plot_double_xy(i_time, var, diff=diff, nTime=n_time, stats=stats, cmap="RdBu_r",
                                  show_plot=False, save=save_dir, vmin=vmin, vmax=vmax)
    else:
        model_desc.plot_double_xy(i_time, var, diff=diff, nTime=n_time, stats=stats, cmap="RdBu_r",
                                  show_plot=False, save=save_dir)


def plot_all_lat_lons(config, i_time, n_time, diff, stats, save_dir):
    argv = ["-c", config]
    setup = SetupDiagnostics(argv)

    print("\nLoading models ...\n")
    models = load_models(setup, skip_causal_phq=True)
    model_key = setup.nn_type

    md = ModelDiagnostics(setup=setup, models=models[model_key])

    dict_keys = models[model_key].keys()

    for var in list(dict_keys):
        print(f"\n\n---- Variable {var}")
        _ = md.plot_double_xy(i_time, var, diff=diff, nTime=n_time, stats=stats, cmap="RdBu_r", show_plot=False,
                              save=save_dir)

        plt.close()
        gc.collect()


if __name__ == "__main__":
    # This main method is mainly used for testing

    # Parameters
    i_time = 1  # 'mean'
    n_time = 5  # 1440  # about a month
    diff = False
    stats = False  # ["r2", "mse"]  # mean, r2

    # Additional params for setting plot color map range
    vmin = False  # False, -3e-7
    vmax = False  # False, 3e-7

    project_root = Path(__file__).parent.parent.resolve()

    base_dir = os.path.join(project_root, "output", "training_30_castle_original")

    config_file = os.path.join(base_dir, "cfg_castle_original.yml")
    plot_dir = os.path.join("plots_offline_evaluation", "debug", "plots_lat_lon")

    variable = "prect"  # prect

    plot_single_variable(variable, config_file, i_time, n_time, diff, stats, vmin, vmax, plot_dir)
