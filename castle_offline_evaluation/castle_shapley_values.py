# Suppress tensorflow info logs
import os

from castle_offline_evaluation.diagnostic_utils import create_castle_model_description

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from pathlib import Path

from neural_networks.load_models import load_single_model
from utils.setup import SetupDiagnostics
from utils.variable import Variable_Lev_Metadata

# Parameters
i_time = 'range'
metric = 'abs_mean'  # 'mean', 'abs_mean', 'abs_mean_sign'
n_time = 100  # 1440 # about month
n_samples = 5  # 1024; 2048; 4096; 8192
project_root = Path(__file__).parent.parent.resolve()

config_file = Path(project_root, "output_castle/training_6_normal/cfg_castle_NN_Creation.yml")
plot_dir = Path(project_root, "output_castle/test_diagnostics/plots_double_xy/")

variable = "prect"


def shap_single_variable(var_name, config):
    argv = ["-c", config]
    setup = SetupDiagnostics(argv)

    var_model = load_single_model(setup, var_name)

    model_desc = create_castle_model_description(setup, var_model)
    return _get_shapley_single_variable(var_name, model_desc)


def _get_shapley_single_variable(var_name, model_desc):
    var = Variable_Lev_Metadata.parse_var_name(var_name)

    return model_desc.get_shapley_values('range', var, nTime=n_time, nSamples=n_samples, metric=metric)


if __name__ == "__main__":
    shap_values_mean, inputs, input_vars_dict = shap_single_variable(variable, config_file)
    print(shap_values_mean)
    print(inputs)
    print(input_vars_dict)
