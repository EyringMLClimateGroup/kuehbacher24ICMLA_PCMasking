import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

def assert_identical_attributes(loaded_model, model):
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


def train_model(model, num_inputs, epochs=2, n_samples=160, batch_size=16, strategy=None):
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


def print_plot_model_summary(model, plot_name, output_dir):
    print(model.summary())
    try:
        keras.utils.plot_model(model, to_file=os.path.join(output_dir, plot_name), show_shapes=True,
                               show_layer_activations=True)
    except ImportError:
        print("WARNING: Cannot plot model because either pydot or graphviz are not installed. "
              "See tf.keras.utils.plot_model documentation for details.")
