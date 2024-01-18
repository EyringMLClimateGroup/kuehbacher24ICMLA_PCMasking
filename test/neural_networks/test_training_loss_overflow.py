import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from neural_networks.castle.building_castle import build_castle
from neural_networks.castle.castle_model_original import CASTLEOriginal
from test.neural_networks.castle.test_castle import train_castle
from test.testing_utils import set_memory_growth_gpu
from utils.setup import SetupNeuralNetworks

# The purpose of these tests is to see whether the training loss is still a number,
# when we have a full network with 94 inputs and 1 outputs. They are separate from normal training tests
# in order to speed up those tests.
#
# Possible options for reducing the training loss:
#  - reduce initial weights
#  - reduce learning rate
#  - adam options: weight decay, clip norm, clip value
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

try:
    set_memory_growth_gpu()
except RuntimeError:
    print("\n\n*** GPU growth could not be enabled. "
          "When running multiple tests, this may be due physical drivers having already been "
          "initialized, in which case memory growth may already be enabled. "
          "If memory growth is not enabled, the tests may fail with CUDA error. ***\n")


@pytest.fixture(scope="module",
                params=["cfg_castle_adapted_all_inputs_outputs_random_normal_notears.yml",
                        "cfg_castle_adapted_all_inputs_outputs_random_uniform_dagma.yml",
                        "cfg_castle_original_all_inputs_outputs_lecun_uniform.yml",
                        "cfg_castle_original_all_inputs_outputs_var_scaling.yml",
                        "cfg_castle_simplified_all_inputs_outputs_random_normal.yml",
                        "cfg_castle_simplified_all_inputs_outputs_random_uniform.yml",
                        "cfg_gumbel_softmax_single_output_model_all_inputs_outputs_random_normal.yml",
                        "cfg_gumbel_softmax_single_output_model_all_inputs_outputs_random_uniform.yml"])
def setup_castle_all_inputs_outputs(request):
    config_file = os.path.join(PROJECT_ROOT, "test", "config", request.param)
    argv = ["-c", config_file]

    return SetupNeuralNetworks(argv)


@pytest.mark.parametrize("strategy", [None, tf.distribute.MirroredStrategy()])
def test_train_castle_overflow(setup_castle_all_inputs_outputs, strategy, seed):
    num_inputs = len(setup_castle_all_inputs_outputs.input_order_list)

    model = build_castle(setup_castle_all_inputs_outputs, num_inputs, setup_castle_all_inputs_outputs.init_lr,
                         eager_execution=True, strategy=strategy, seed=seed)

    epochs = 1
    if isinstance(model, CASTLEOriginal):
        network_inputs = num_inputs + 1
    else:
        network_inputs = num_inputs

    batch_size = 1024
    n_samples = batch_size * 100
    history = train_castle(model=model, num_inputs=network_inputs, epochs=epochs, n_samples=n_samples,
                           batch_size=batch_size, strategy=strategy)

    assert (isinstance(history, tf.keras.callbacks.History))

    assert (history.history["loss"][-1] is not np.inf)


@pytest.mark.parametrize("strategy", [None, tf.distribute.MirroredStrategy()])
def test_train_castle_stress_test(setup_castle_all_inputs_outputs, strategy, seed):
    num_inputs = len(setup_castle_all_inputs_outputs.input_order_list)

    model = build_castle(setup_castle_all_inputs_outputs, num_inputs, setup_castle_all_inputs_outputs.init_lr,
                         eager_execution=True, strategy=strategy, seed=seed)

    epochs = 1
    if isinstance(model, CASTLEOriginal):
        network_inputs = num_inputs + 1
    else:
        network_inputs = num_inputs

    batch_size = 8192 * 4
    n_samples = batch_size * 100
    history = train_castle(model=model, num_inputs=network_inputs, epochs=epochs, n_samples=n_samples,
                           batch_size=batch_size, strategy=strategy)

    assert (isinstance(history, tf.keras.callbacks.History))

    assert (history.history["loss"][-1] is not np.inf)
