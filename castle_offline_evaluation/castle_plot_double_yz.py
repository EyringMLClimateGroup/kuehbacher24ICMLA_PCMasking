# Suppress tensorflow info logs
import os

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from castle_offline_evaluation.castle_evaluation_utils import create_castle_model_description

from pathlib import Path

from neural_networks.load_models import load_single_model, load_models
from neural_networks.model_diagnostics import ModelDiagnostics
from utils.setup import SetupDiagnostics
from utils.variable import Variable_Lev_Metadata
import matplotlib.pyplot as plt
import gc


def plot_single_variable(var_name, config, i_time, n_time, ilon, stats, vmin, vmax, save_dir):
    argv = ["-c", config]
    setup = SetupDiagnostics(argv)

    var_model = load_single_model(setup, var_name)

    model_desc = create_castle_model_description(setup, var_model)
    var = Variable_Lev_Metadata.parse_var_name(var_name)

    if var.var.dimensions != 3:
        raise ValueError(f"Can only plot vertical cross-section for 3d variables. "
                         f"Given variable {var.var.value} has dimension {var.var.dimensions}.")
    var_keys = _get_var_keys(var.var.value, var_model.keys())

    if vmax and vmin:
        model_desc.plot_double_yz(var, var_keys, itime=i_time, nTime=n_time, ilon=ilon, diff=False,
                                  cmap='RdBu_r', save=save_dir, stats=stats, vmin=vmin, vmax=vmax)
    else:
        model_desc.plot_double_yz(var, var_keys, itime=i_time, nTime=n_time, ilon=ilon, diff=False,
                                  cmap='RdBu_r', save=save_dir, stats=stats)


def _get_var_keys(var_str, dict_keys):
    return [k for k in dict_keys if var_str in str(k)]


def plot_all_cross_sections(i_time, n_time, i_lon, diff, stats, config, plot_dir):
    argv = ["-c", config]
    setup = SetupDiagnostics(argv)

    print("\nLoading models ...\n")
    models = load_models(setup, skip_causal_phq=True)
    model_key = setup.nn_type

    md = ModelDiagnostics(setup=setup, models=models[model_key])

    three_d_str = ["tphystnd-3.64", "phq-3.64"]
    three_d_keys = [Variable_Lev_Metadata.parse_var_name(var_str) for var_str in three_d_str]

    dict_keys = models[model_key].keys()

    print("\nComputing cross section plots ...\n")
    for var in three_d_keys:
        print(f"\n\n---- Variable {var}")
        var_keys = [k for k in dict_keys if var.var.value in str(k)]

        _ = md.plot_double_yz(var, var_keys, itime=i_time, nTime=n_time, ilon=i_lon, diff=diff,
                              cmap='RdBu_r', stats=stats, show_plot=False, save=plot_dir)
        plt.close()
        gc.collect()


if __name__ == "__main__":
    # Parameters
    i_time = 1  # 1, 'mean', 'range' --> range doesn't work
    n_time = 1  # 1440 (about a month), 5855, False
    stats = ["mse", "r2"]  # False, 'r2'
    ilon = 64  # 64, 'mean'

    # Additional params for setting plot color map range
    vmin = False  # False, -3e-7
    vmax = False  # False, 3e-7

    project_root = Path(__file__).parent.parent.resolve()

    config_file = Path(project_root,
                       "output_castle/training_28_custom_mirrored_functional/cfg_castle_training_run_2.yml")
    plot_dir = Path(project_root,
                    "output_castle/training_28_custom_mirrored_functional/plots_offline_evaluation/debug/plots_cross_section/")

    variable = "tphystnd-0"  # tphystnd-0, phq-0 (any level will do here, but it has to be specified)
    # variables = ["tphystnd-0", "phq-0"]

    plot_single_variable(variable, config_file, i_time, n_time, ilon, stats, vmin, vmax, plot_dir)
    # plot_multiple_variables(variables, config_file, i_time, n_time, ilon, stats, vim, vmax, plot_dir)
