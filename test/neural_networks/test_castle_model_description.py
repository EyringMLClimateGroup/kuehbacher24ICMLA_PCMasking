import os

import pytest
from mock import patch

from neural_networks.load_models import load_models
from neural_networks.models import generate_models, ModelDescription
from neural_networks.training import train_all_models
from test.testing_utils import delete_output_dirs, set_memory_growth_gpu, train_model_if_not_exists, set_strategy

try:
    set_memory_growth_gpu()
except RuntimeError:
    print("\n\n*** GPU growth could not be enabled. "
          "When running multiple tests, this may be due physical drivers having already been "
          "initialized, in which case memory growth may already be enabled. "
          "If memory growth is not enabled, the tests may fail with CUDA error. ***\n")


@pytest.mark.parametrize("strategy", ["", "mirrored"])
@pytest.mark.parametrize("setup_str", ["setup_castle_adapted_2d_dagma", "setup_castle_adapted_2d_notears",
                                       "setup_castle_adapted_w3d",
                                       "setup_castle_original_2d", "setup_castle_original_w3d",
                                       "setup_gumbel_softmax_single_output_model_2d",
                                       "setup_gumbel_softmax_single_output_model_w3d"])
def test_create_castle_model_description(setup_str, strategy, request):
    setup = request.getfixturevalue(setup_str)
    setup = set_strategy(setup, strategy)

    model_descriptions = generate_models(setup)

    assert (isinstance(model_descriptions, list))

    # Check number of models
    if "2d" in setup_str:
        num_models = 2
    elif "w3d" in setup_str:
        num_models = 31
    assert (len(model_descriptions) == num_models)

    for m in model_descriptions:
        assert (isinstance(m, ModelDescription))


@pytest.mark.parametrize("setup_str", ["setup_castle_adapted_2d_dagma", "setup_castle_adapted_2d_notears",
                                       "setup_castle_adapted_w3d",
                                       "setup_castle_original_2d", "setup_castle_original_w3d",
                                       "setup_gumbel_softmax_single_output_model_2d",
                                       "setup_gumbel_softmax_single_output_model_w3d"])
@patch("neural_networks.models.tf.config.get_visible_devices")
def test_create_castle_model_description_distributed_value_error(mocked_visible_devices, setup_str, request):
    setup = request.getfixturevalue(setup_str)

    # Mock that there aren't any visible devices
    mocked_visible_devices.return_value = []

    setup.distribute_strategy = "mirrored"

    with pytest.raises(EnvironmentError):
        _ = generate_models(setup)


@pytest.mark.parametrize("strategy", ["", "mirrored"])
@pytest.mark.parametrize("setup_str", ["setup_castle_adapted_2d_dagma", "setup_castle_adapted_2d_notears",
                                       "setup_castle_original_2d", "setup_gumbel_softmax_single_output_model_2d"])
def test_train_and_save_castle_model_description(setup_str, strategy, request):
    setup = request.getfixturevalue(setup_str)
    setup = set_strategy(setup, strategy)

    model_descriptions = generate_models(setup)
    delete_output_dirs(model_descriptions, setup)

    train_all_models(model_descriptions, setup)

    # Assert output exists
    for m in model_descriptions:
        model_fn = m.get_filename() + "_model.keras"
        out_path = str(m.get_path(setup.nn_output_path))

        assert (os.path.isfile(os.path.join(out_path, model_fn)))
        assert (os.path.isdir(setup.tensorboard_folder))


@pytest.mark.parametrize("strategy", ["", "mirrored"])
@pytest.mark.parametrize("setup_str", ["setup_castle_adapted_2d_dagma", "setup_castle_adapted_2d_notears",
                                       "setup_castle_original_2d", "setup_gumbel_softmax_single_output_model_2d"])
def test_load_castle_model_description(setup_str, strategy, request):
    setup = request.getfixturevalue(setup_str)

    setup.distribute_strategy = strategy

    train_model_if_not_exists(setup)
    loaded_model_description = load_models(setup)

    assert (len(loaded_model_description[setup.nn_type]) == len(setup.output_order))
