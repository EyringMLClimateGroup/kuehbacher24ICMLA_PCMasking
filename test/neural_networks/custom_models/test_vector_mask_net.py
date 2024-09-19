import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from pcmasking.neural_networks.custom_models.building_custom_model import build_custom_model
from pcmasking.neural_networks.custom_models.mask_model import MaskNet
from test.neural_networks.custom_models.utils import assert_identical_attributes, train_model, create_dataset, \
    print_plot_model_summary
from test.testing_utils import set_memory_growth_gpu, generate_output_var_list

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
print(PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test", "output", "test_mask_net")
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
@pytest.mark.parametrize("setup_str", ["setup_mask_net_2d", "setup_mask_net_w3d",
                                       "setup_mask_net_2d_threshold_file",
                                       "setup_mask_net_w3d_threshold_file"])
def test_create_mask_net(setup_str, strategy, seed, request):
    setup = request.getfixturevalue(setup_str)
    num_inputs = len(setup.input_order_list)

    var = generate_output_var_list(setup)[0] if "threshold_file" in setup_str else None
    model = build_custom_model(setup, num_inputs, setup.init_lr, output_var=var,
                               eager_execution=False, strategy=strategy, seed=seed)

    assert (isinstance(model, MaskNet))
    assert (isinstance(model.outputs, list))
    assert (len(model.outputs[0].shape) == 2)
    assert (model.outputs[0].shape[-1] == 1)
    print_plot_model_summary(model, setup_str + ".png", OUTPUT_DIR)


@pytest.mark.parametrize("strategy", [None, tf.distribute.MirroredStrategy()])
@pytest.mark.parametrize("setup_str", ["setup_mask_net_2d", "setup_mask_net_w3d",
                                       "setup_mask_net_2d_threshold_file",
                                       "setup_mask_net_w3d_threshold_file"])
def test_train_mask_net(setup_str, strategy, seed, request):
    setup = request.getfixturevalue(setup_str)
    num_inputs = len(setup.input_order_list)

    var = generate_output_var_list(setup)[0] if "threshold_file" in setup_str else None
    model = build_custom_model(setup, num_inputs, setup.init_lr, output_var=var,
                               eager_execution=False, strategy=strategy, seed=seed)

    epochs = 2
    network_inputs = num_inputs
    history = train_model(model, network_inputs, epochs=epochs, strategy=strategy)

    assert (isinstance(history, tf.keras.callbacks.History))

    train_loss_keys = [m.name for m in model.metric_dict.values()]
    val_loss_keys = ["val_" + loss for loss in train_loss_keys]
    assert (all(k in history.history.keys() for k in train_loss_keys))
    assert (all(k in history.history.keys() for k in val_loss_keys))

    assert (len(history.history["loss"]) == epochs)


@pytest.mark.parametrize("strategy", [None, tf.distribute.MirroredStrategy()])
@pytest.mark.parametrize("setup_str", ["setup_mask_net_2d", "setup_mask_net_w3d",
                                       "setup_mask_net_2d_threshold_file",
                                       "setup_mask_net_w3d_threshold_file"])
def test_predict_mask_net(setup_str, strategy, seed, request):
    setup = request.getfixturevalue(setup_str)
    num_inputs = len(setup.input_order_list)

    var = generate_output_var_list(setup)[0] if "threshold_file" in setup_str else None
    model = build_custom_model(setup, num_inputs, setup.init_lr, output_var=var,
                               eager_execution=False, strategy=strategy, seed=seed)

    n_samples = 160
    batch_size = 16

    network_inputs = num_inputs
    test_ds = create_dataset(network_inputs, n_samples=n_samples, batch_size=batch_size, strategy=strategy)
    prediction = model.predict(test_ds)

    num_batches = int(n_samples / batch_size)

    assert (prediction is not None)
    assert (prediction.shape == (batch_size * num_batches, 1))


@pytest.mark.parametrize("strategy", [None, tf.distribute.MirroredStrategy()])
@pytest.mark.parametrize("setup_str", ["setup_mask_net_2d", "setup_mask_net_w3d",
                                       "setup_mask_net_2d_threshold_file",
                                       "setup_mask_net_w3d_threshold_file"])
def test_save_load_mask_net(setup_str, strategy, seed, request):
    setup = request.getfixturevalue(setup_str)
    num_inputs = len(setup.input_order_list)

    var = generate_output_var_list(setup)[0] if "threshold_file" in setup_str else None
    model = build_custom_model(setup, num_inputs, setup.init_lr, output_var=var,
                               eager_execution=False, strategy=strategy, seed=seed)

    _ = train_model(model, num_inputs, epochs=1, strategy=strategy)

    model_save_name = "model_" + setup_str + ".keras"
    weights_save_name = "model_" + setup_str + "_weights.h5"
    model.save(os.path.join(OUTPUT_DIR, model_save_name), save_format="keras_v3")
    model.save_weights(os.path.join(OUTPUT_DIR, weights_save_name))

    loaded_model = tf.keras.models.load_model(os.path.join(OUTPUT_DIR, model_save_name),
                                              custom_objects={"MaskNet": MaskNet})

    assert (loaded_model.threshold == model.threshold)
    assert (np.all(loaded_model.masking_vector.numpy() == model.masking_vector.numpy()))

    assert_identical_attributes(loaded_model, model)
