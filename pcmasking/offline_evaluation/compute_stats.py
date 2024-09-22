import datetime
import os
import pickle
import time
from pathlib import Path

from pcmasking.neural_networks.load_models import load_models
from pcmasking.neural_networks.model_diagnostics import ModelDiagnostics
from pcmasking.utils.setup import SetupDiagnostics


def compute_stats(i_time, n_time, config, plot_dir):
    argv = ["-c", config]
    setup = SetupDiagnostics(argv)

    print("\nLoading models ...\n")
    models = load_models(setup, skip_causal_phq=True)
    model_key = setup.nn_type

    md = ModelDiagnostics(setup=setup, models=models[model_key])
    dict_keys = models[model_key].keys()

    save_dir = Path(plot_dir, get_save_str(i_time, num_time=n_time))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    stats_dict = dict()

    print("\nComputing stats ...\n")
    for var in dict_keys:
        print(f"\n\n---- Variable {var}")
        md.compute_stats(i_time, var, nTime=n_time)
        stats_dict[str(var)] = md.stats

    print("\nFinished computing stats. Saving stats ...\n")
    f_name = f"hor_stats.p"
    out_file = os.path.join(save_dir, f_name)
    with open(out_file, "wb") as f:
        pickle.dump(stats_dict, f)

    print(f"\nSaved stats file {Path(*Path(out_file).parts[-5:])}\n\n")


def get_save_str(idx_time, num_time=False):
    if type(idx_time) is int:
        idx_time_str = f"step-{idx_time}"
    elif type(idx_time) is str:
        if num_time:
            idx_time_str = f"{idx_time}-{num_time}"
        else:
            idx_time_str = f"{idx_time}-all"
    else:
        raise ValueError(f"Unkown value for idx_time: {idx_time}")

    return idx_time_str


if __name__ == "__main__":
    # This main method is mainly used for testing
    ##########################################
    # Parameters
    i_time = "range"
    n_time = 1440

    project_root = Path(__file__).parent.parent.parent.resolve()

    training_dir = Path("models/mask_net")
    config_file = os.path.join(project_root, training_dir, "cfg_mask_net_thresholds_train.yml")
    plot_dir = os.path.join(project_root, "plots_offline_evaluation/debug/stats/")
    ##########################################

    print(f"\n\n{datetime.datetime.now()} --- Start computing stats.", flush=True)
    t_init = time.time()

    compute_stats(i_time, n_time, config_file, plot_dir)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
