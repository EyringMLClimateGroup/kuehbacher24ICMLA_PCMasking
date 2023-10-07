# Suppress tensorflow info logs
import os

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from castle_offline_evaluation.castle_evaluation_utils import create_castle_model_description

from pathlib import Path

from neural_networks.load_models import load_single_model
from utils.setup import SetupDiagnostics
from utils.variable import Variable_Lev_Metadata

# Parameters
i_time = 1  # 1, 'mean', 'range' --> range doesn't work
n_time = 1440  # 1440 (about a month), 5855, False
stats = False  # False, 'r2'
ilon = 64  # 64, 'mean'

# Additional params for setting plot color map range
vmin = False  # False, -3e-7
vmax = False  # False, 3e-7

project_root = Path(__file__).parent.parent.resolve()

config_file = Path(project_root, "output_castle/training_7_mirrored_dict/cfg_castle_NN_Creation.yml")
plot_dir = Path(project_root, "output_castle/test_diagnostics/plots_double_yz/")

variable = "tphystnd-0"  # tphystnd-0, phq-0 (any level will do here, but it has to be specified)
variables = ["tphystnd-0", "phq-0"]


def plot_single_variable(var_name, config, save_dir):
    argv = ["-c", config]
    setup = SetupDiagnostics(argv)

    var_model = load_single_model(setup, var_name)

    model_desc = create_castle_model_description(setup, var_model)
    _plot_single_variable_double_yz(var_name, model_desc, var_model.keys(), save_dir)


def _plot_single_variable_double_yz(var_name, model_desc, dict_keys, save_dir):
    var = Variable_Lev_Metadata.parse_var_name(var_name)

    if var.var.dimensions != 3:
        raise ValueError(f"Can only plot vertical cross-section for 3d variables. "
                         f"Given variable {var.var.value} has dimension {var.var.dimensions}.")
    var_keys = _get_var_keys(var.var.value, dict_keys)

    if vmax and vmin:
        model_desc.plot_double_yz(var, var_keys, itime=i_time, nTime=n_time, ilon=ilon, diff=False,
                                  cmap='RdBu_r', save=save_dir, stats=stats, vmin=vmin, vmax=vmax)
    else:
        model_desc.plot_double_yz(var, var_keys, itime=i_time, nTime=n_time, ilon=ilon, diff=False,
                                  cmap='RdBu_r', save=save_dir, stats=stats)


def _get_var_keys(var_str, dict_keys):
    return [k for k in dict_keys if var_str in str(k)]


def plot_multiple_variables(var_names, config, save_dir):
    argv = ["-c", config]
    setup = SetupDiagnostics(argv)

    var_models = dict()
    for var_name in variables:
        var_models.update(load_single_model(setup, var_name))

    model_desc = create_castle_model_description(setup, var_models)
    _plot_multiple_vars_double_xy(var_names, model_desc, var_models.keys(), save_dir)


def _plot_multiple_vars_double_xy(var_names, model_desc, dict_keys, save_dir):
    vars = [Variable_Lev_Metadata.parse_var_name(var_name) for var_name in var_names]

    var_keys_all = {var: _get_var_keys(var.var.value, dict_keys) for var in vars}

    for var in vars:
        if var.var.dimensions != 3:
            raise ValueError(f"Can only plot vertical cross-section for 3d variables. "
                             f"Given variable {var.var.value} has dimension {var.var.dimensions}.")
        if vmax and vmin:
            model_desc.plot_double_yz(var, var_keys_all[var], itime=i_time, nTime=n_time, ilon=ilon, diff=False,
                                      cmap='RdBu_r', save=save_dir, stats=stats, vmin=vmin, vmax=vmax)
        else:
            model_desc.plot_double_yz(var, var_keys_all[var], itime=i_time, nTime=n_time, ilon=ilon, diff=False,
                                      cmap='RdBu_r', save=save_dir, stats=stats)


if __name__ == "__main__":
    plot_single_variable(variable, config_file, plot_dir)
    # plot_multiple_variables(variables, config_file, plot_dir)
