import pytest

from neural_networks.load_models import load_models
from neural_networks.model_diagnostics import ModelDiagnostics
from pathlib import Path
from test.testing_utils import train_model_if_not_exists
import os
from utils.setup import SetupDiagnostics

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


@pytest.fixture()
def diagnostic_setup_castle_adapted_2d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_castle_adapted_2d_eval_train.yml")
    argv = ["-c", config_file]

    return SetupDiagnostics(argv)


@pytest.fixture()
def diagnostic_setup_castle_original_2d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_castle_original_2d_eval_train.yml")
    argv = ["-c", config_file]

    return SetupDiagnostics(argv)


@pytest.fixture()
def diagnostic_setup_castle_simplified_2d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config",
                               "cfg_castle_simplified_2d_eval_train.yml")
    argv = ["-c", config_file]

    return SetupDiagnostics(argv)


@pytest.fixture()
def diagnostic_setup_gumbel_softmax_single_output_model_2d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config",
                               "cfg_gumbel_softmax_single_output_model_2d_eval_train.yml")
    argv = ["-c", config_file]

    return SetupDiagnostics(argv)


@pytest.fixture()
def castle_model_original(diagnostic_setup_castle_original_2d):
    train_model_if_not_exists(diagnostic_setup_castle_original_2d)
    return load_models(diagnostic_setup_castle_original_2d)


@pytest.fixture()
def castle_model_adapted(diagnostic_setup_castle_adapted_2d):
    train_model_if_not_exists(diagnostic_setup_castle_adapted_2d)
    return load_models(diagnostic_setup_castle_adapted_2d)


@pytest.fixture()
def castle_model_simplified(diagnostic_setup_castle_simplified_2d):
    train_model_if_not_exists(diagnostic_setup_castle_simplified_2d)
    return load_models(diagnostic_setup_castle_simplified_2d)


@pytest.fixture()
def gumbel_softmax_single_output_model(diagnostic_setup_gumbel_softmax_single_output_model_2d):
    train_model_if_not_exists(diagnostic_setup_gumbel_softmax_single_output_model_2d)
    return load_models(diagnostic_setup_gumbel_softmax_single_output_model_2d)


@pytest.fixture()
def castle_model_description_original(diagnostic_setup_castle_original_2d, castle_model_original):
    diagnostic_setup_castle_original_2d.model_type = diagnostic_setup_castle_original_2d.nn_type
    diagnostic_setup_castle_original_2d.use_val_batch_size = False
    return ModelDiagnostics(setup=diagnostic_setup_castle_original_2d,
                            models=castle_model_original[diagnostic_setup_castle_original_2d.nn_type])


@pytest.fixture()
def castle_model_description_simplified(diagnostic_setup_castle_simplified_2d, castle_model_simplified):
    diagnostic_setup_castle_simplified_2d.model_type = diagnostic_setup_castle_simplified_2d.nn_type
    diagnostic_setup_castle_simplified_2d.use_val_batch_size = False
    return ModelDiagnostics(setup=diagnostic_setup_castle_simplified_2d,
                            models=castle_model_simplified[diagnostic_setup_castle_simplified_2d.nn_type])


@pytest.fixture()
def castle_model_description_adapted(diagnostic_setup_castle_adapted_2d, castle_model_adapted):
    diagnostic_setup_castle_adapted_2d.model_type = diagnostic_setup_castle_adapted_2d.nn_type
    diagnostic_setup_castle_adapted_2d.use_val_batch_size = False
    return ModelDiagnostics(setup=diagnostic_setup_castle_adapted_2d,
                            models=castle_model_adapted[diagnostic_setup_castle_adapted_2d.nn_type])


@pytest.fixture()
def model_description_gumbel_softmax_multi_output(diagnostic_setup_gumbel_softmax_single_output_model_2d,
                                                  gumbel_softmax_single_output_model):
    diagnostic_setup_gumbel_softmax_single_output_model_2d.model_type = diagnostic_setup_gumbel_softmax_single_output_model_2d.nn_type
    diagnostic_setup_gumbel_softmax_single_output_model_2d.use_val_batch_size = False
    return ModelDiagnostics(setup=diagnostic_setup_gumbel_softmax_single_output_model_2d,
                            models=gumbel_softmax_single_output_model[
                                diagnostic_setup_gumbel_softmax_single_output_model_2d.nn_type])
