# Suppress tensorflow info logs
import os

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from castle_offline_evaluation.castle_evaluation_utils import create_castle_model_description

from pathlib import Path

from neural_networks.load_models import load_single_model
from utils.setup import SetupDiagnostics
from utils.variable import Variable_Lev_Metadata


def plot_single_variable(var_name, config, i_time, n_time, diff, stats, vmin, vmax, save_dir):
    argv = ["-c", config]
    setup = SetupDiagnostics(argv)

    var_model = load_single_model(setup, var_name)

    model_desc = create_castle_model_description(setup, var_model)

    var = Variable_Lev_Metadata.parse_var_name(var_name)
    if vmax and vmin:
        model_desc.plot_double_xy(i_time, var, diff=diff, nTime=n_time, stats=stats, cmap="RdBu_r",
                                  show_plot=False, save=save_dir, vmin=vmin, vmax=vmax)
    else:
        model_desc.plot_double_xy(i_time, var, diff=diff, nTime=n_time, stats=stats, cmap="RdBu_r",
                                  show_plot=False, save=save_dir)


def plot_multiple_variables(var_names, config, i_time, n_time, diff, stats, vmin, vmax, save_dir):
    argv = ["-c", config]
    setup = SetupDiagnostics(argv)

    var_models = dict()
    for var_name in var_names:
        var_models.update(load_single_model(setup, var_name))

    model_desc = create_castle_model_description(setup, var_models)

    variables = [Variable_Lev_Metadata.parse_var_name(var_name) for var_name in var_names]
    if vmin and vmax:
        for var in variables:
            model_desc.plot_double_xy(i_time, var, diff=diff, nTime=n_time, stats=stats, cmap="RdBu_r",
                                      show_plot=False, save=save_dir, vmin=vmin, vmax=vmax)
    else:
        for var in variables:
            model_desc.plot_double_xy(i_time, var, diff=diff, nTime=n_time, stats=stats, cmap="RdBu_r",
                                      show_plot=False, save=save_dir)


if __name__ == "__main__":
    # Parameters
    i_time = 1  # 'mean', 'range' --> range doesn't work
    n_time = 1440  # about a month
    diff = False
    stats = ["r2", "mse"]  # mean, r2

    # Additional params for setting plot color map range
    vmin = False  # False, -3e-7
    vmax = False  # False, 3e-7

    project_root = Path(__file__).parent.parent.resolve()

    config_file = Path(project_root, "output_castle/training_28_custom_mirrored_functional/cfg_castle_training_run_2.yml")
    plot_dir = Path(project_root,
                    "output_castle/training_28_custom_mirrored_functional/plots_offline_evaluation/debug/plots_lat_lon/")

    variable = "prect"  # prect

    plot_single_variable(variable, config_file, i_time, n_time, diff, stats, vmin, vmax, plot_dir)
