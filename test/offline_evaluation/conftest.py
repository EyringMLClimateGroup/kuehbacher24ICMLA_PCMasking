import os
from pathlib import Path

import pytest

from pcmasking.neural_networks.load_models import load_models
from pcmasking.neural_networks.model_diagnostics import ModelDiagnostics
from test.testing_utils import train_model_if_not_exists, create_masking_vector, generate_output_var_list
from pcmasking.utils.setup import SetupDiagnostics

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


@pytest.fixture()
def diagnostic_setup_pre_mask_net_2d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config",
                               "cfg_pre_mask_net_2d_eval_train.yml")
    argv = ["-c", config_file]

    return SetupDiagnostics(argv)


@pytest.fixture()
def diagnostic_setup_mask_net_2d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_mask_net_2d_eval_train.yml")
    argv = ["-c", config_file]

    setup = SetupDiagnostics(argv)

    num_inputs = len(setup.input_order_list)
    create_masking_vector(num_inputs, setup.masking_vector_file, outputs_list=generate_output_var_list(setup))

    return setup


@pytest.fixture()
def pre_mask_net_model(diagnostic_setup_pre_mask_net_2d):
    train_model_if_not_exists(diagnostic_setup_pre_mask_net_2d)
    return load_models(diagnostic_setup_pre_mask_net_2d)


@pytest.fixture()
def mask_net_model(diagnostic_setup_mask_net_2d):
    train_model_if_not_exists(diagnostic_setup_mask_net_2d)
    return load_models(diagnostic_setup_mask_net_2d)


@pytest.fixture()
def model_description_pre_mask_net(diagnostic_setup_pre_mask_net_2d, pre_mask_net_model):
    diagnostic_setup_pre_mask_net_2d.model_type = diagnostic_setup_pre_mask_net_2d.nn_type
    diagnostic_setup_pre_mask_net_2d.use_val_batch_size = False
    return ModelDiagnostics(setup=diagnostic_setup_pre_mask_net_2d,
                            models=pre_mask_net_model[diagnostic_setup_pre_mask_net_2d.nn_type])


@pytest.fixture()
def model_description_mask_net(diagnostic_setup_mask_net_2d, mask_net_model):
    diagnostic_setup_mask_net_2d.model_type = diagnostic_setup_mask_net_2d.nn_type
    diagnostic_setup_mask_net_2d.use_val_batch_size = False
    return ModelDiagnostics(setup=diagnostic_setup_mask_net_2d,
                            models=mask_net_model[diagnostic_setup_mask_net_2d.nn_type])
