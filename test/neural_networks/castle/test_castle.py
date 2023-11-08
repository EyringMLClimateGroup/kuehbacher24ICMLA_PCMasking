import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from neural_networks.castle.building_castle import build_castle
from neural_networks.castle.castle_model_original import CASTLEOriginal
from neural_networks.castle.castle_model_adapted import CASTLEAdapted
from neural_networks.castle.masked_dense_layer import MaskedDenseLayer
from test.testing_utils import set_memory_growth_gpu

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
print(PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test", "output", "test_castle")
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
@pytest.mark.parametrize("setup", ["setup_castle_adapted_2d_dagma", "setup_castle_adapted_2d_notears",
                                   "setup_castle_adapted_w3d"])
def test_create_castle_adapted(setup, strategy, seed, request):
    setup = request.getfixturevalue(setup)
    num_inputs = len(setup.input_order_list)

    model = build_castle(setup, num_inputs, setup.init_lr,
                         eager_execution=True, strategy=strategy, seed=seed)

    assert (isinstance(model, CASTLEAdapted))
    assert (isinstance(model.outputs, list))
    assert (len(model.outputs[0].shape) == 3)
    _print_plot_model_summary(model)


@pytest.mark.parametrize("strategy", [None, tf.distribute.MirroredStrategy()])
@pytest.mark.parametrize("setup", ["setup_castle_adapted_2d_dagma", "setup_castle_adapted_2d_notears",
                                   "setup_castle_original_w3d"])
def test_create_castle_original(setup, strategy, seed, request):
    setup = request.getfixturevalue(setup)
    num_inputs = len(setup.input_order_list)

    model = build_castle(setup, num_inputs, setup.init_lr,
                         eager_execution=True, strategy=strategy, seed=seed)

    assert (isinstance(model, CASTLEOriginal))
    assert (isinstance(model.outputs, list))
    assert (len(model.outputs[0].shape) == 3)
    _print_plot_model_summary(model)


def test_create_castle_adapted_multiple_lr_kernel_init(setup_castle_adapted_multiple_lr_kernel_init, seed):
    num_inputs = len(setup_castle_adapted_multiple_lr_kernel_init.input_order_list)

    model = build_castle(setup_castle_adapted_multiple_lr_kernel_init, num_inputs,
                         setup_castle_adapted_multiple_lr_kernel_init.init_lr,
                         eager_execution=True, strategy=None, seed=seed)

    assert (isinstance(model, CASTLEAdapted))


def _print_plot_model_summary(model):
    print(model.summary())
    try:
        keras.utils.plot_model(model, to_file=os.path.join(OUTPUT_DIR, "castle_adapted_vars_2d.png"), show_shapes=True,
                               show_layer_activations=True)
    except ImportError:
        print("WARNING: Cannot plot model because either pydot or graphviz are not installed. "
              "See tf.keras.utils.plot_model documentation for details.")


@pytest.mark.parametrize("strategy", [None, tf.distribute.MirroredStrategy()])
@pytest.mark.parametrize("setup_str", ["setup_castle_adapted_2d_dagma", "setup_castle_adapted_2d_notears",
                                       "setup_castle_adapted_w3d",
                                       "setup_castle_original_2d", "setup_castle_original_w3d"])
def test_train_castle(setup_str, strategy, seed, request):
    setup = request.getfixturevalue(setup_str)
    num_inputs = len(setup.input_order_list)

    model = build_castle(setup, num_inputs, setup.init_lr,
                         eager_execution=True, strategy=strategy, seed=seed)

    epochs = 2
    if "original" in setup_str:
        network_inputs = num_inputs + 1
    else:
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
                                       "setup_castle_adapted_w3d",
                                       "setup_castle_original_2d", "setup_castle_original_w3d"])
def test_predict_castle(setup_str, strategy, seed, request):
    setup = request.getfixturevalue(setup_str)
    num_inputs = len(setup.input_order_list)

    model = build_castle(setup, num_inputs, setup.init_lr,
                         eager_execution=True, strategy=strategy, seed=seed)

    n_samples = 160
    batch_size = 16

    if "original" in setup_str:
        network_inputs = num_inputs + 1
    else:
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

    model = build_castle(setup, num_inputs, setup.init_lr,
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

    _assert_identical_attributes(loaded_model, model)


@pytest.mark.parametrize("strategy", [None, tf.distribute.MirroredStrategy()])
@pytest.mark.parametrize("setup_str", ["setup_castle_adapted_2d_dagma", "setup_castle_adapted_2d_notears",
                                       "setup_castle_original_w3d"])
def test_save_load_castle_original(setup_str, strategy, seed, request):
    setup = request.getfixturevalue(setup_str)
    num_inputs = len(setup.input_order_list)

    model = build_castle(setup, num_inputs, setup.init_lr,
                         eager_execution=True, strategy=strategy, seed=seed)

    _ = train_castle(model, num_inputs + 1, epochs=1, strategy=strategy)

    model_save_name = "model_" + setup_str + ".keras"
    weights_save_name = "model_" + setup_str + "_weights.h5"
    model.save(os.path.join(OUTPUT_DIR, model_save_name), save_format="keras_v3")
    model.save_weights(os.path.join(OUTPUT_DIR, weights_save_name))

    loaded_model = tf.keras.models.load_model(os.path.join(OUTPUT_DIR, model_save_name),
                                              custom_objects={"CASTLEOriginal": CASTLEOriginal,
                                                              "MaskedDenseLayer": MaskedDenseLayer})

    assert (loaded_model.beta == model.beta)
    assert (loaded_model.lambda_weight == model.lambda_weight)

    _assert_identical_attributes(loaded_model, model)


def _assert_identical_attributes(loaded_model, model):
    assert (loaded_model.alpha == model.alpha)
    assert (loaded_model.rho == model.rho)
    assert (loaded_model.activation == model.activation)

    assert (type(loaded_model.kernel_initializer_input_layers) == type(model.kernel_initializer_input_layers))
    assert (type(loaded_model.kernel_initializer_hidden_layers) == type(model.kernel_initializer_hidden_layers))
    assert (type(loaded_model.kernel_initializer_output_layers) == type(model.kernel_initializer_output_layers))

    assert (type(loaded_model.bias_initializer_input_layers) == type(model.bias_initializer_input_layers))
    assert (type(loaded_model.bias_initializer_hidden_layers) == type(model.bias_initializer_hidden_layers))
    assert (type(loaded_model.bias_initializer_output_layers) == type(model.bias_initializer_output_layers))

    assert (type(loaded_model.kernel_regularizer_input_layers) == type(model.kernel_regularizer_input_layers))
    assert (type(loaded_model.kernel_regularizer_hidden_layers) == type(model.kernel_regularizer_hidden_layers))
    assert (type(loaded_model.kernel_regularizer_output_layers) == type(model.kernel_regularizer_output_layers))

    assert (type(loaded_model.bias_regularizer_input_layers) == type(model.bias_regularizer_input_layers))
    assert (type(loaded_model.bias_regularizer_hidden_layers) == type(model.bias_regularizer_hidden_layers))
    assert (type(loaded_model.bias_regularizer_output_layers) == type(model.bias_regularizer_output_layers))

    assert (type(loaded_model.activity_regularizer_input_layers) == type(model.activity_regularizer_input_layers))
    assert (type(loaded_model.activity_regularizer_hidden_layers) == type(model.activity_regularizer_hidden_layers))
    assert (type(loaded_model.activity_regularizer_output_layers) == type(model.activity_regularizer_output_layers))

    assert (loaded_model.relu_alpha == model.relu_alpha)

    assert (loaded_model.seed == model.seed)

    assert (len(loaded_model.get_weights()) == len(model.get_weights()))


def train_castle(model, num_inputs, epochs=2, n_samples=160, batch_size=16, strategy=None):
    train_ds = create_dataset(num_inputs, n_samples=n_samples, batch_size=batch_size, strategy=strategy)
    val_ds = create_dataset(num_inputs, n_samples=n_samples, batch_size=batch_size, strategy=strategy)

    history = model.fit(
        x=train_ds,
        validation_data=val_ds,
        batch_size=batch_size,
        epochs=epochs
    )

    return history


def create_dataset(num_inputs, n_samples=160, batch_size=16, strategy=None):
    num_outputs = 1

    x_array = np.random.standard_normal((n_samples, num_inputs)).astype(dtype=np.float32)
    y_array = np.random.standard_normal((n_samples, num_outputs)).astype(dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((x_array, y_array)).batch(batch_size, drop_remainder=True)

    if strategy is not None:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        dataset = dataset.with_options(options)

    return dataset
