import os
import sys
from pathlib import Path

# This is necessary when calling this script not from the root
# project_root = Path(__file__).parent.parent.resolve()
# sys.path.append(str(project_root))


# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from utils.setup import SetupDiagnostics
from neural_networks.load_models import load_models
from neural_networks.model_diagnostics import ModelDiagnostics
from utils.variable import Variable_Lev_Metadata


def get_save_str(idx_time, num_time=False, idx_lon=False,
                 show_diff=False, statistics=False):
    if type(idx_time) is int:
        idx_time_str = f"step-{idx_time}"
    elif type(idx_time) is str:
        if num_time:
            idx_time_str = f"{idx_time}-{num_time}"
        else:
            idx_time_str = f"{idx_time}-all"
    else:
        raise ValueError(f"Unkown value for idx_time: {idx_time}")

    idx_lon_str = f"_ilon-{idx_lon}" if idx_lon else ""
    stats_str = f"_stats-{statistics}" if statistics else ""
    diff_str = "_with_diff" if show_diff else "_no_diff"

    return idx_time_str + idx_lon_str + stats_str + diff_str


def run_plot_yz(variables, md_var_keys, i_time, n_time, i_lon, diff, stats):
    print(f"\nPlotting {i_time}-{n_time}-{i_lon}-{diff}-{stats}", flush=True)

    save_dir = Path(plot_dir, get_save_str(i_time, num_time=n_time, idx_lon=i_lon,
                                           show_diff=diff, statistics=stats))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for var in variables:
        print(var)
        var_keys = [k for k in md_var_keys if var.var.value in str(k)]

        _ = castle_md.plot_double_yz(var, var_keys, itime=i_time, nTime=n_time, ilon=i_lon, diff=diff,
                                     cmap='RdBu_r', stats=stats, show_plot=False, save=save_dir)


if __name__ == "__main__":
    print("\nCreating cross-section plots for 3d variables tphystnd and phq.")
    print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    project_root = Path(__file__).parent.parent.resolve()

    argv = ["-c", Path(project_root, "output_castle/training_7_mirrored/cfg_castle_NN_Creation.yml")]
    plot_dir = Path(project_root, "output_castle/training_7_mirrored/plots_offline_evaluation/plots_cross_section/")

    castle_setup = SetupDiagnostics(argv)
    castle_models = load_models(castle_setup)

    # This variable does not exist in the code (but key nn_type is the same)
    castle_model_type = "castleNN"
    castle_setup.model_type = castle_model_type

    castle_md = ModelDiagnostics(setup=castle_setup,
                                 models=castle_models[castle_model_type])

    dict_keys = castle_models['castleNN'].keys()

    # only 3d
    three_d_str = ["tphystnd-3.64", "phq-3.64"]
    three_d_keys = [Variable_Lev_Metadata.parse_var_name(var_str) for var_str in three_d_str]

    # time step 1 without diff
    # i_time = 1
    # n_time = False
    # i_lon = 64
    # diff = False
    # stats = False
    # run_plot_yz(three_d_keys, dict_keys, i_time, n_time, i_lon, diff, stats)
    #
    # # time step 1 with diff
    # i_time = 1
    # n_time = False
    # i_lon = 64
    # diff = True
    # stats = False
    # run_plot_yz(three_d_keys, dict_keys, i_time, n_time, i_lon, diff, stats)

    # time mean without diff
    i_time = "mean"
    n_time = 1440
    i_lon = 64
    diff = False
    stats = False
    run_plot_yz(three_d_keys, dict_keys, i_time, n_time, i_lon, diff, stats)

    # time mean with diff
    i_time = "mean"
    n_time = 1440
    i_lon = 64
    diff = True
    stats = False
    run_plot_yz(three_d_keys, dict_keys, i_time, n_time, i_lon, diff, stats)

    # time mean, lon mean, stats r2, no diff
    i_time = "mean"
    n_time = 1440 # 720
    i_lon = "mean"
    diff = False
    stats = "r2"
    run_plot_yz(three_d_keys, dict_keys, i_time, n_time, i_lon, diff, stats)

    # time mean, lon mean, stats r2, with diff
    i_time = "mean"
    n_time = 1440
    i_lon = "mean"
    diff = True
    stats = "r2"
    run_plot_yz(three_d_keys, dict_keys, i_time, n_time, i_lon, diff, stats)

    # time mean, lon mean, stats mse, no diff
    i_time = "mean"
    n_time = 1440
    i_lon = "mean"
    diff = False
    stats = "mse"
    run_plot_yz(three_d_keys, dict_keys, i_time, n_time, i_lon, diff, stats)

    # time mean, lon mean, stats mse, with diff
    i_time = "mean"
    n_time = 1440
    i_lon = "mean"
    diff = True
    stats = "mse"
    run_plot_yz(three_d_keys, dict_keys, i_time, n_time, i_lon, diff, stats)
