import os
from pathlib import Path

import pytest
import tensorflow as tf

from neural_networks.custom_models.building_custom_model import build_custom_model
from neural_networks.custom_models.castle_model_adapted import CASTLEAdapted
from neural_networks.custom_models.layers.masked_dense_layer import MaskedDenseLayer
from test.neural_networks.custom_models.utils import assert_identical_attributes, train_castle, create_dataset, \
    print_plot_model_summary
from test.testing_utils import set_memory_growth_gpu

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
print(PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test", "output", "test_castle_adapted")
print(OUTPUT_DIR)

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

try:
    set_memory_growth_gpu()
except RuntimeError:
    print("\n\n*** GPU growth could not be enabled. "
          "When running multiple tests, this may be due physical drivers having already been "
          "initialized, in which case memory growth may already be enabled. "
          "If memory growth is not enabled, the tests may fail with CUDA error. ***\n")


@pytest.mark.parametrize("strategy", [None, tf.distribute.MirroredStrategy()])
@pytest.mark.parametrize("setup_str", ["setup_castle_adapted_2d_dagma", "setup_castle_adapted_2d_notears",
                                       "setup_castle_adapted_w3d"])
def test_create_castle_adapted(setup_str, strategy, seed, request):
    setup = request.getfixturevalue(setup_str)
    num_inputs = len(setup.input_order_list)

    model = build_custom_model(setup, num_inputs, setup.init_lr,
                               eager_execution=True, strategy=strategy, seed=seed)

    assert (isinstance(model, CASTLEAdapted))
    assert (isinstance(model.outputs, list))
    assert (len(model.outputs[0].shape) == 3)
    print_plot_model_summary(model, setup_str + ".png", OUTPUT_DIR)


def test_create_castle_adapted_multiple_lr_kernel_init(setup_castle_adapted_multiple_lr_kernel_init, seed):
    num_inputs = len(setup_castle_adapted_multiple_lr_kernel_init.input_order_list)

    model = build_custom_model(setup_castle_adapted_multiple_lr_kernel_init, num_inputs,
                               setup_castle_adapted_multiple_lr_kernel_init.init_lr,
                               eager_execution=True, strategy=None, seed=seed)

    assert (isinstance(model, CASTLEAdapted))


@pytest.mark.parametrize("strategy", [None, tf.distribute.MirroredStrategy()])
@pytest.mark.parametrize("setup_str", ["setup_castle_adapted_2d_dagma", "setup_castle_adapted_2d_notears",
                                       "setup_castle_adapted_w3d"])
def test_train_castle_adapted(setup_str, strategy, seed, request):
    setup = request.getfixturevalue(setup_str)
    num_inputs = len(setup.input_order_list)

    model = build_custom_model(setup, num_inputs, setup.init_lr,
                               eager_execution=True, strategy=strategy, seed=seed)

    epochs = 2
    network_inputs = num_inputs
    history = train_castle(model, network_inputs, epochs=epochs, strategy=strategy)

    assert (isinstance(history, tf.keras.callbacks.History))

    train_loss_keys = [m.name for m in model.metric_dict.values()]
    val_loss_keys = ["val_" + loss for loss in train_loss_keys]
    assert (all(k in history.history.keys() for k in train_loss_keys))
    assert (all(k in history.history.keys() for k in val_loss_keys))

    assert (len(history.history["loss"]) == epochs)


@pytest.mark.parametrize("strategy", [None, tf.distribute.MirroredStrategy()])
@pytest.mark.parametrize("setup_str", ["setup_castle_adapted_2d_dagma", "setup_castle_adapted_2d_notears",
                                       "setup_castle_adapted_w3d"])
def test_predict_castle_adapted(setup_str, strategy, seed, request):
    setup = request.getfixturevalue(setup_str)
    num_inputs = len(setup.input_order_list)

    model = build_custom_model(setup, num_inputs, setup.init_lr,
                               eager_execution=True, strategy=strategy, seed=seed)

    n_samples = 160
    batch_size = 16

    network_inputs = num_inputs
    test_ds = create_dataset(network_inputs, n_samples=n_samples, batch_size=batch_size, strategy=strategy)
    prediction = model.predict(test_ds)

    num_batches = int(n_samples / batch_size)

    assert (prediction is not None)
    assert (prediction.shape == (batch_size * num_batches, num_inputs + 1, 1))


@pytest.mark.parametrize("strategy", [None, tf.distribute.MirroredStrategy()])
@pytest.mark.parametrize("setup_str", ["setup_castle_adapted_2d_dagma", "setup_castle_adapted_2d_notears",
                                       "setup_castle_adapted_w3d"])
def test_save_load_castle_adapted(setup_str, strategy, seed, request):
    setup = request.getfixturevalue(setup_str)
    num_inputs = len(setup.input_order_list)

    model = build_custom_model(setup, num_inputs, setup.init_lr,
                               eager_execution=True, strategy=strategy, seed=seed)

    _ = train_castle(model, num_inputs, epochs=1, strategy=strategy)

    model_save_name = "model_" + setup_str + ".keras"
    weights_save_name = "model_" + setup_str + "_weights.h5"
    model.save(os.path.join(OUTPUT_DIR, model_save_name), save_format="keras_v3")
    model.save_weights(os.path.join(OUTPUT_DIR, weights_save_name))

    loaded_model = tf.keras.models.load_model(os.path.join(OUTPUT_DIR, model_save_name),
                                              custom_objects={"CASTLEAdapted": CASTLEAdapted,
                                                              "MaskedDenseLayer": MaskedDenseLayer})

    assert (loaded_model.lambda_prediction == model.lambda_prediction)
    assert (loaded_model.lambda_sparsity == model.lambda_sparsity)
    assert (loaded_model.lambda_reconstruction == model.lambda_reconstruction)
    assert (loaded_model.lambda_acyclicity == model.lambda_acyclicity)
    assert (loaded_model.relu_alpha == model.relu_alpha)
    assert (loaded_model.acyclicity_constraint == model.acyclicity_constraint)

    assert (loaded_model.alpha == model.alpha)
    assert (loaded_model.rho == model.rho)

    assert_identical_attributes(loaded_model, model)
