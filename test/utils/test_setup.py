import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from utils.setup import SetupNeuralNetworks
from utils.variable import SPCAM_Vars

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


@pytest.mark.parametrize("config_name", ["cfg_castle_original_2d.yml", "cfg_castle_original_w3d.yml"])
def test_create_setup_castle_original(config_name):
    config_file = os.path.join(PROJECT_ROOT, "test", "config", config_name)
    argv = ["-c", config_file]

    setup = SetupNeuralNetworks(argv)

    # Assert
    with open(config_file, "r") as f:
        yml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    assert (setup.nn_type == "CASTLEOriginal")

    assert (setup.beta == yml_cfg["beta"])
    assert (setup.lambda_weight == yml_cfg["lambda_weight"])

    assert (setup.rho == yml_cfg["rho"])
    assert (setup.alpha == yml_cfg["alpha"])

    _assert_identical_attributes(setup, yml_cfg)


@pytest.mark.parametrize("config_name", ["cfg_castle_adapted_2d_dagma.yml", "cfg_castle_adapted_2d_notears.yml",
                                         "cfg_castle_adapted_w3d.yml"])
def test_create_setup_castle_adapted(config_name):
    config_file = os.path.join(PROJECT_ROOT, "test", "config", config_name)
    argv = ["-c", config_file]

    setup = SetupNeuralNetworks(argv)

    # Assert
    with open(config_file, "r") as f:
        yml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    assert (setup.nn_type == "CASTLEAdapted")

    assert (setup.lambda_prediction == yml_cfg["lambda_prediction"])
    assert (setup.lambda_sparsity == yml_cfg["lambda_sparsity"])
    assert (setup.lambda_acyclicity == yml_cfg["lambda_acyclicity"])
    assert (setup.lambda_reconstruction == yml_cfg["lambda_reconstruction"])

    assert (setup.rho == yml_cfg["rho"])
    assert (setup.alpha == yml_cfg["alpha"])

    assert (setup.acyclicity_constraint == yml_cfg["acyclicity_constraint"])

    _assert_identical_attributes(setup, yml_cfg)


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

    _assert_identical_attributes(setup, yml_cfg)



@pytest.mark.parametrize("config_name", ["cfg_vector_mask_net_2d.yml", "cfg_vector_mask_net_w3d.yml",
                                         "cfg_vector_mask_net_2d_threshold_file.yml",
                                         "cfg_vector_mask_net_w3d_threshold_file.yml"])
def test_create_setup_vector_mask_net(config_name):
    config_file = os.path.join(PROJECT_ROOT, "test", "config", config_name)
    argv = ["-c", config_file]

    setup = SetupNeuralNetworks(argv)

    # Assert
    with open(config_file, "r") as f:
        yml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    assert (setup.nn_type == "VectorMaskNet")
    assert ((Path(*Path(setup.masking_vector_file).parts[-4:])) == Path(yml_cfg["masking_vector_file"]))

    if "threshold_file" in config_name:
        assert ((Path(*Path(setup.mask_threshold_file).parts[-4:])) == Path(yml_cfg["mask_threshold_file"]))
    else:
        assert (setup.mask_threshold == yml_cfg["mask_threshold"])


@pytest.mark.parametrize("config_name", ["cfg_gumbel_softmax_single_output_model_2d.yml",
                                         "cfg_gumbel_softmax_single_output_model_w3d.yml"])
def test_create_setup_gumbel_softmax_single_output_model(config_name):
    config_file = os.path.join(PROJECT_ROOT, "test", "config", config_name)
    argv = ["-c", config_file]

    setup = SetupNeuralNetworks(argv)

    # Assert
    with open(config_file, "r") as f:
        yml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    assert (setup.nn_type == "GumbelSoftmaxSingleOutputModel")
    assert (setup.lambda_prediction == yml_cfg["lambda_prediction"])
    assert (setup.lambda_crf == yml_cfg["lambda_crf"])
    assert (setup.lambda_vol_min == yml_cfg["lambda_vol_min"])
    assert (setup.lambda_vol_avg == yml_cfg["lambda_vol_avg"])

    assert (setup.sigma_crf == yml_cfg["sigma_crf"])
    assert (np.all(setup.level_bins == yml_cfg["level_bins"]))

    assert (setup.temperature == yml_cfg["temperature"])

    assert (setup.temperature_decay_rate == yml_cfg["temperature_decay_rate"])
    assert (setup.temperature_decay_steps == yml_cfg["temperature_decay_steps"])
    try:
        assert (setup.temperature_warm_up == yml_cfg["temperature_warm_up"])
    except KeyError:
        assert (setup.temperature_warm_up == 0)

    _assert_identical_attributes(setup, yml_cfg)


@pytest.mark.parametrize("config_name",
                         ["cfg_castle_adapted_2d_lr_cosine_init_orthogonal_random_normal_random_uniform.yml",
                          "cfg_castle_adapted_2d_lr_exp_init_he_normal_he_uniform_identity.yml",
                          "cfg_castle_adapted_2d_lr_linear_init_lecun_normal_lecun_uniform_ones.yml",
                          "cfg_castle_adapted_2d_lr_none_init_constant_glorot_normal_glorot_uniform.yml",
                          "cfg_castle_adapted_2d_lr_plateau_init_none_none_none.yml",
                          "cfg_castle_adapted_2d_lr_plateau_init_trunc_normal_var_scaling_zeros.yml"])
def test_create_setup_castle_lr_schedule_kernel_initializer(config_name):
    config_file = os.path.join(PROJECT_ROOT, "test", "config", config_name)
    argv = ["-c", config_file]

    setup = SetupNeuralNetworks(argv)

    # Assert
    with open(config_file, "r") as f:
        yml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Assert learning rate schedule
    if "lr_none" in config_name or "lr_exp" in config_name:
        assert (setup.lr_schedule["schedule"] == "exponential")
        assert (setup.lr_schedule["step"] == yml_cfg["step_lr"])
        assert (setup.lr_schedule["divide"] == yml_cfg["divide_lr"])
    elif "lr_plateau" in config_name:
        assert (setup.lr_schedule["schedule"] == "plateau")
        assert (setup.lr_schedule["monitor"] == yml_cfg["monitor"])
        assert (setup.lr_schedule["factor"] == yml_cfg["factor"])
        assert (setup.lr_schedule["min_lr"] == float(yml_cfg["min_lr"]))
        assert (setup.lr_schedule["patience"] == yml_cfg["patience"])
    if "lr_linear" in config_name:
        assert (setup.lr_schedule["schedule"] == "linear")
        assert (setup.lr_schedule["decay_steps"] == yml_cfg["decay_steps"])
        assert (setup.lr_schedule["end_lr"] == float(yml_cfg["end_lr"]))
    elif "lr_cosine" in config_name:
        assert (setup.lr_schedule["schedule"] == "cosine")
        assert (setup.lr_schedule["decay_steps"] == yml_cfg["decay_steps"])
        assert (setup.lr_schedule["alpha"] == yml_cfg["cosine_alpha"])
        assert (setup.lr_schedule["warmup_steps"] == yml_cfg["warmup_steps"])

    # Assert kernel initializer
    if "init_orthogonal_random_normal_random_uniform" in config_name:
        assert (setup.kernel_initializer_input_layers["initializer"] == "Orthogonal")
        assert (setup.kernel_initializer_input_layers["gain"] == yml_cfg["input_init_orthogonal_gain"])
        assert (setup.kernel_initializer_hidden_layers["initializer"] == "RandomNormal")
        assert (setup.kernel_initializer_hidden_layers["mean"] == yml_cfg["hidden_init_random_normal_mean"])
        assert (setup.kernel_initializer_hidden_layers["std"] == yml_cfg["hidden_init_random_normal_std"])
        assert (setup.kernel_initializer_output_layers["initializer"] == "RandomUniform")
        assert (setup.kernel_initializer_output_layers["min_val"] == yml_cfg[
            "output_init_random_uniform_min_val"])
        assert (setup.kernel_initializer_output_layers["max_val"] == yml_cfg[
            "output_init_random_uniform_max_val"])

    if "init_he_normal_he_uniform_identity" in config_name:
        assert (setup.kernel_initializer_input_layers["initializer"] == "HeNormal")
        assert (setup.kernel_initializer_hidden_layers["initializer"] == "HeUniform")
        assert (setup.kernel_initializer_output_layers["initializer"] == "Identity")
        assert (setup.kernel_initializer_output_layers["gain"] == yml_cfg["output_init_identity_gain"])

    if "init_lecun_normal_lecun_uniform_ones" in config_name:
        assert (setup.kernel_initializer_input_layers["initializer"] == "LecunNormal")
        assert (setup.kernel_initializer_hidden_layers["initializer"] == "LecunUniform")
        assert (setup.kernel_initializer_output_layers["initializer"] == "Ones")

    if "init_constant_glorot_normal_glorot_uniform" in config_name:
        assert (setup.kernel_initializer_input_layers["initializer"] == "Constant")
        assert (setup.kernel_initializer_input_layers["value"] == yml_cfg["input_init_constant_value"])
        assert (setup.kernel_initializer_hidden_layers["initializer"] == "GlorotNormal")
        assert (setup.kernel_initializer_output_layers["initializer"] == "GlorotUniform")

    if "init_trunc_normal_var_scaling_zeros" in config_name:
        assert (setup.kernel_initializer_input_layers["initializer"] == "TruncatedNormal")
        assert (setup.kernel_initializer_input_layers["mean"] == yml_cfg["input_init_truncated_normal_mean"])
        assert (setup.kernel_initializer_input_layers["std"] == yml_cfg["input_init_truncated_normal_std"])
        assert (setup.kernel_initializer_hidden_layers["initializer"] == "VarianceScaling")
        assert (setup.kernel_initializer_hidden_layers["scale"] == yml_cfg["hidden_init_variance_scaling_scale"])
        assert (setup.kernel_initializer_hidden_layers["mode"] == yml_cfg["hidden_init_variance_scaling_mode"])
        assert (setup.kernel_initializer_hidden_layers["distribution"] == yml_cfg[
            "hidden_init_variance_scaling_distribution"])
        assert (setup.kernel_initializer_output_layers["initializer"] == "Zeros")

    if "init_none_none_none" in config_name:
        assert (setup.kernel_initializer_input_layers is None)
        assert (setup.kernel_initializer_hidden_layers is None)
        assert (setup.kernel_initializer_output_layers is None)


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
    assert (isinstance(setup.kernel_initializer_input_layers, dict))
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
