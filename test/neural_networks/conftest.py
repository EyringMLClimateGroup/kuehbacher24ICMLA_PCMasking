import os
from pathlib import Path

import pytest

from test.testing_utils import create_masking_vector, generate_output_var_list
from utils.setup import SetupNeuralNetworks

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


# default time out
def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker('timeout') is None:
            item.add_marker(pytest.mark.timeout(180))


@pytest.fixture()
def setup_castle_adapted_2d_dagma():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_castle_adapted_2d_dagma.yml")
    argv = ["-c", config_file]

    return SetupNeuralNetworks(argv)


@pytest.fixture()
def setup_castle_adapted_2d_notears():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_castle_adapted_2d_notears.yml")
    argv = ["-c", config_file]

    return SetupNeuralNetworks(argv)


@pytest.fixture()
def setup_castle_adapted_w3d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_castle_adapted_w3d.yml")
    argv = ["-c", config_file]

    return SetupNeuralNetworks(argv)


@pytest.fixture()
def setup_castle_original_2d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_castle_original_2d.yml")
    argv = ["-c", config_file]

    return SetupNeuralNetworks(argv)


@pytest.fixture()
def setup_castle_original_w3d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_castle_original_w3d.yml")
    argv = ["-c", config_file]

    return SetupNeuralNetworks(argv)


@pytest.fixture()
def setup_castle_simplified_2d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_castle_simplified_2d.yml")
    argv = ["-c", config_file]

    return SetupNeuralNetworks(argv)


@pytest.fixture()
def setup_castle_simplified_w3d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_castle_simplified_w3d.yml")
    argv = ["-c", config_file]

    return SetupNeuralNetworks(argv)


@pytest.fixture()
def setup_gumbel_softmax_single_output_model_2d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_gumbel_softmax_single_output_model_2d.yml")
    argv = ["-c", config_file]

    return SetupNeuralNetworks(argv)


@pytest.fixture()
def setup_gumbel_softmax_single_output_model_w3d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_gumbel_softmax_single_output_model_w3d.yml")
    argv = ["-c", config_file]

    return SetupNeuralNetworks(argv)


@pytest.fixture()
def setup_vector_mask_net_2d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_vector_mask_net_2d.yml")
    argv = ["-c", config_file]

    setup = SetupNeuralNetworks(argv)

    num_inputs = len(setup.input_order_list)
    create_masking_vector(num_inputs, setup.masking_vector_file, outputs_list=generate_output_var_list(setup))

    return setup


@pytest.fixture()
def setup_vector_mask_net_w3d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_vector_mask_net_w3d.yml")
    argv = ["-c", config_file]

    setup = SetupNeuralNetworks(argv)

    num_inputs = len(setup.input_order_list)
    create_masking_vector(num_inputs, setup.masking_vector_file, outputs_list=generate_output_var_list(setup))

    return setup


@pytest.fixture(params=["cfg_castle_adapted_2d_lr_cosine_init_orthogonal_random_normal_random_uniform.yml",
                        "cfg_castle_adapted_2d_lr_exp_init_he_normal_he_uniform_identity.yml",
                        "cfg_castle_adapted_2d_lr_linear_init_lecun_normal_lecun_uniform_ones.yml",
                        "cfg_castle_adapted_2d_lr_none_init_constant_glorot_normal_glorot_uniform.yml",
                        "cfg_castle_adapted_2d_lr_plateau_init_none_none_none.yml",
                        "cfg_castle_adapted_2d_lr_plateau_init_trunc_normal_var_scaling_zeros.yml"])
def setup_castle_adapted_multiple_lr_kernel_init(request):
    config_file = os.path.join(PROJECT_ROOT, "test", "config", request.param)
    argv = ["-c", config_file]

    return SetupNeuralNetworks(argv)


@pytest.fixture()
def seed():
    return 42
