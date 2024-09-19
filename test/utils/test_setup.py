import os
from pathlib import Path

import pytest
import yaml

from pcmasking.utils.setup import SetupNeuralNetworks
from pcmasking.utils.variable import SPCAM_Vars

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


@pytest.mark.parametrize("config_name", ["cfg_pre_mask_net_2d.yml",
                                         "cfg_pre_mask_net_w3d.yml"])
def test_create_setup_pre_mask_net(config_name):
    config_file = os.path.join(PROJECT_ROOT, "test", "config", config_name)
    argv = ["-c", config_file]

    setup = SetupNeuralNetworks(argv)

    # Assert
    with open(config_file, "r") as f:
        yml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    assert (setup.nn_type == "PreMaskNet")
    assert (setup.lambda_sparsity == yml_cfg["lambda_sparsity"])

    assert (isinstance(setup.kernel_initializer_input_layers, dict))

    _assert_identical_attributes(setup, yml_cfg)


@pytest.mark.parametrize("config_name", ["cfg_mask_net_2d.yml", "cfg_mask_net_w3d.yml",
                                         "cfg_mask_net_2d_threshold_file.yml",
                                         "cfg_mask_net_w3d_threshold_file.yml"])
def test_create_setup_mask_net(config_name):
    config_file = os.path.join(PROJECT_ROOT, "test", "config", config_name)
    argv = ["-c", config_file]

    setup = SetupNeuralNetworks(argv)

    # Assert
    with open(config_file, "r") as f:
        yml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    assert (setup.nn_type == "MaskNet")
    assert ((Path(*Path(setup.masking_vector_file).parts[-4:])) == Path(yml_cfg["masking_vector_file"]))

    if "threshold_file" in config_name:
        assert ((Path(*Path(setup.mask_threshold_file).parts[-4:])) == Path(yml_cfg["mask_threshold_file"]))
    else:
        assert (setup.mask_threshold == yml_cfg["mask_threshold"])

    # There is no input layer kernel in MaskNet
    assert (setup.kernel_initializer_input_layers is None)

    _assert_identical_attributes(setup, yml_cfg)


def _assert_identical_attributes(setup, yml_cfg):
    if yml_cfg["activation"].lower() == "leakyrelu":
        try:
            assert (setup.relu_alpha == yml_cfg["relu_alpha"])
        except KeyError:
            # default value in setup for relu_alpha is 0.3
            assert (setup.relu_alpha == 0.3)

    assert (setup.distribute_strategy == yml_cfg["distribute_strategy"])

    assert (isinstance(setup.additional_val_datasets, list))
    assert (isinstance(setup.additional_val_datasets[0], dict))

    assert (isinstance(setup.lr_schedule, dict))
    assert (isinstance(setup.kernel_initializer_hidden_layers, dict))
    assert (isinstance(setup.kernel_initializer_output_layers, dict))

    input_vars = [var for var in SPCAM_Vars if var.name in yml_cfg["spcam_parents"]]
    output_vars = [var for var in SPCAM_Vars if var.name in yml_cfg["spcam_children"]]

    assert (_compare_spcam_var_lists(input_vars, setup.spcam_inputs))
    assert (_compare_spcam_var_lists(output_vars, setup.spcam_outputs))


def _compare_spcam_var_lists(s, t):
    t = list(t)  # make a mutable copy
    try:
        for elem in s:
            t.remove(elem)
    except ValueError:
        return False
    return not t
