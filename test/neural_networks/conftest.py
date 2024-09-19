import os
from pathlib import Path

import pytest

from test.testing_utils import create_masking_vector, generate_output_var_list, create_threshold_file
from pcmasking.utils.setup import SetupNeuralNetworks

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


# default time out
def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker('timeout') is None:
            item.add_marker(pytest.mark.timeout(180))


@pytest.fixture()
def setup_pre_mask_net_2d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_pre_mask_net_2d.yml")
    argv = ["-c", config_file]

    return SetupNeuralNetworks(argv)


@pytest.fixture()
def setup_pre_mask_net_w3d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_pre_mask_net_w3d.yml")
    argv = ["-c", config_file]

    return SetupNeuralNetworks(argv)


@pytest.fixture()
def setup_mask_net_2d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_mask_net_2d.yml")
    argv = ["-c", config_file]

    setup = SetupNeuralNetworks(argv)

    num_inputs = len(setup.input_order_list)

    create_masking_vector(num_inputs, setup.masking_vector_file, outputs_list=generate_output_var_list(setup))

    return setup


@pytest.fixture()
def setup_mask_net_2d_threshold_file():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_mask_net_2d_threshold_file.yml")
    argv = ["-c", config_file]

    setup = SetupNeuralNetworks(argv)

    num_inputs = len(setup.input_order_list)
    outputs_list = generate_output_var_list(setup)

    create_masking_vector(num_inputs, setup.masking_vector_file, outputs_list=outputs_list)
    create_threshold_file(setup.mask_threshold_file, outputs_list=outputs_list)

    return setup


@pytest.fixture()
def setup_mask_net_w3d():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_mask_net_w3d.yml")
    argv = ["-c", config_file]

    setup = SetupNeuralNetworks(argv)

    num_inputs = len(setup.input_order_list)
    create_masking_vector(num_inputs, setup.masking_vector_file, outputs_list=generate_output_var_list(setup))

    return setup


@pytest.fixture()
def setup_mask_net_w3d_threshold_file():
    config_file = os.path.join(PROJECT_ROOT, "test", "config", "cfg_mask_net_w3d_threshold_file.yml")
    argv = ["-c", config_file]

    setup = SetupNeuralNetworks(argv)

    num_inputs = len(setup.input_order_list)
    outputs_list = generate_output_var_list(setup)

    create_masking_vector(num_inputs, setup.masking_vector_file, outputs_list=outputs_list)
    create_threshold_file(setup.mask_threshold_file, outputs_list=outputs_list)

    return setup


@pytest.fixture()
def seed():
    return 42
