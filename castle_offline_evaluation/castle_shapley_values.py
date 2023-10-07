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

from castle_offline_evaluation.castle_evaluation_utils import create_castle_model_description, read_txt_to_dict

from pathlib import Path
import pickle

from neural_networks.load_models import load_single_model
from utils.setup import SetupDiagnostics
from utils.variable import Variable_Lev_Metadata


def shap_single_variable(var_name, config, n_time, n_samples, metric):
    argv = ["-c", config]
    setup = SetupDiagnostics(argv)

    var_model = load_single_model(setup, var_name)

    model_desc = create_castle_model_description(setup, var_model)

    var = Variable_Lev_Metadata.parse_var_name(var_name)
    return model_desc.get_shapley_values("range", var, nTime=n_time, nSamples=n_samples, metric=metric)


def get_save_str(idx_time, num_time=False, num_samples=False, shap_metric=False):
    if type(idx_time) is int:
        idx_time_str = f"step-{idx_time}"
    elif type(idx_time) is str:
        if num_time:
            idx_time_str = f"{idx_time}-{num_time}"
        else:
            idx_time_str = f"{idx_time}-all"
    else:
        raise ValueError(f"Unkown value for idx_time: {idx_time}")

    samples_str = f"_samples-{num_samples}" if num_samples else "_samples-all"
    metric_str = f"_{shap_metric}" if shap_metric else ""

    return idx_time_str + samples_str + metric_str


def save_shapley_dict(out_path, var, shap_values, inputs, map_dict):
    out_file = "shap_values_" + map_dict[str(var)] + ".p"
    shap_dict = dict()

    shap_dict["inputs"] = inputs
    shap_dict["shap_values_mean"] = shap_values

    with open(os.path.join(out_path, out_file), 'wb') as f:
        pickle.dump(shap_dict, f)


if __name__ == "__main__":
    ##########################################
    # Parameters
    i_time = 'range'
    metric = 'abs_mean'  # 'mean', 'abs_mean', 'abs_mean_sign'
    n_time = 5  # 1440 # about month
    n_samples = 5  # 1024; 2048; 4096; 8192
    project_root = Path(__file__).parent.parent.resolve()

    config_file = Path(project_root, "output_castle/training_28_custom_mirrored_functional/cfg_castle_training.yml")
    plot_dir = Path(project_root, "output_castle/training_28_custom_mirrored_functional/shapley/")
    outputs_map = Path("../output_castle/training_28_custom_mirrored_functional/outputs_map.txt")

    variable = "tphystnd-691.39"
    ##########################################

    print(f"\n\n{datetime.datetime.now()} --- Start computing Shapley values for variable {variable}.", flush=True)
    t_init = time.time()
    shap_values, inputs, input_vars_dict = shap_single_variable(variable, config_file, n_time, n_samples, metric)

    shap_dict = dict()
    shap_dict["inputs"] = inputs
    shap_dict["shap_values_mean"] = shap_values

    map_dict = read_txt_to_dict(outputs_map)
    save_dir = Path(plot_dir, get_save_str(i_time, num_time=n_time, num_samples=n_samples, shap_metric=metric))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    save_shapley_dict(save_dir, Variable_Lev_Metadata.parse_var_name(variable), shap_values, inputs, map_dict)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
