import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from pcmasking.neural_networks.custom_models.mask_model import MaskNet
from pcmasking.neural_networks.custom_models.pre_mask_model import PreMaskNet
from pcmasking.utils.variable import Variable_Lev_Metadata


def build_custom_model(setup, num_x_inputs, learning_rate=0.001, output_var=None, eager_execution=False, strategy=None,
                       seed=None):
    """
    Builds and compiles a custom neural network model.

    Args:
        setup (utils.setup.SetupNeuralNetworks): A utils.setup.SetupNeuralNetworks instance containing
            specifics for model creation (e.g. number of hidden layers, activation function, etc)
        num_x_inputs (int): The number of regressors, i.e. the x-variables.
        learning_rate (float): Optimizer learning rate: Defaults to 0.001.
        eager_execution (bool): If `True`, the code will be executed in eager mode and the model's logic will
            not be wrapped inside a tf.function. Can be used for debugging purposes. Defaults to `False`.
        strategy (tf.distribute.Strategy): State and compute distribution policy for parallelization
            across GPUs and/or SLURM nodes.
        seed (int): Random seed. Used to make the behavior of the initializer deterministic.
            Note that a seeded initializer will produce the same random values across multiple calls.

    Returns:
        tf.keras.Model: A tf.keras model.

    Raises:
        ValueError: If the `setup.nn_type` is not in `['PreMaskNet', 'MaskNet']`.
    """
    # Enable eager execution for debugging
    tf.config.run_functions_eagerly(eager_execution)
    # Force eager execution of tf.data functions as well
    if eager_execution:
        tf.data.experimental.enable_debug_mode()

    _set_kernel_initializer(setup)

    try:
        relu_alpha = setup.relu_alpha
    except AttributeError:
        relu_alpha = None

    def _build_custom_model():
        # Build model

        if setup.nn_type == "PreMaskNet":
            model_ = PreMaskNet(num_x_inputs, setup.hidden_layers, setup.activation,
                                lambda_sparsity=setup.lambda_sparsity,
                                relu_alpha=relu_alpha, seed=seed,
                                kernel_initializer_input_layers=setup.kernel_initializer_input_layers,
                                kernel_initializer_hidden_layers=setup.kernel_initializer_hidden_layers,
                                kernel_initializer_output_layers=setup.kernel_initializer_output_layers)

        elif setup.nn_type == "MaskNet":
            # Get masking vector for output variable
            if output_var is not None:
                setup.masking_vector_file = Path(str(setup.masking_vector_file).format(var=output_var))

            print(f"\nLoading masking vector {(Path(*Path(setup.masking_vector_file).parts[-4:]))}\n")
            masking_vector = np.load(setup.masking_vector_file)

            # Get threshold for masking vector
            if setup.mask_threshold is not None:
                # Single float threshold given
                threshold = setup.mask_threshold
            elif setup.mask_threshold_file is not None:
                # Make sure that the output variable is given
                if output_var is None:
                    raise ValueError("Must pass output variable of type Variable_Lev_Metadata when providing "
                                     "threshold in threshold file for MaskNet.")

                # Dictionary with threshold values per output variable given
                print(f"\nLoading threshold file {(Path(*Path(setup.mask_threshold_file).parts[-4:]))}\n")

                with open(setup.mask_threshold_file, "rb") as in_file:
                    mask_threshold_file = pickle.load(in_file)
                    threshold = mask_threshold_file[output_var]
            else:
                raise ValueError(f"Neither mask threshold float value or threshold file has been given.")

            print(f"\nUsing threshold {threshold}\n")
            model_ = MaskNet(num_x_inputs, setup.hidden_layers, setup.activation, masking_vector,
                             threshold=threshold, relu_alpha=relu_alpha, seed=seed,
                             kernel_initializer_hidden_layers=setup.kernel_initializer_hidden_layers,
                             kernel_initializer_output_layers=setup.kernel_initializer_output_layers)

        else:
            raise ValueError(f"Unknown custom model type {setup.nn_type}. Must be one of ['PreMaskNet', 'MaskNet'].")

        model_.build(input_shape=(None, num_x_inputs))
        # Compile model
        return _compile_model(model_, learning_rate, eager_execution)

    if strategy is not None:
        with strategy.scope():
            model = _build_custom_model()
    else:
        model = _build_custom_model()
    return model


def _compile_model(model, learning_rate, eager_execution):
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
        jit_compile=True
    )

    model.compile(
        optimizer=optimizer,
        run_eagerly=eager_execution
    )

    return model


def _set_kernel_initializer(setup):
    if setup.nn_type == "PreMaskNet":
        if setup.kernel_initializer_input_layers is None:
            setup.kernel_initializer_input_layers = {"initializer": "RandomNormal",
                                                     "mean": 0.0,
                                                     "std": 0.01}

        if setup.kernel_initializer_hidden_layers is None:
            setup.kernel_initializer_hidden_layers = {"initializer": "GlorotUniform"}

        if setup.kernel_initializer_output_layers is None:
            setup.kernel_initializer_output_layers = {"initializer": "GlorotUniform"}

    elif setup.nn_type == "MaskNet":
        # No input layer kernel Initializer
        if setup.kernel_initializer_hidden_layers is None:
            setup.kernel_initializer_hidden_layers = {"initializer": "GlorotUniform"}

        if setup.kernel_initializer_output_layers is None:
            setup.kernel_initializer_output_layers = {"initializer": "GlorotUniform"}

    else:
        raise ValueError(f"Unknown custom model type {setup.nn_type}. Must be one of ['PreMaskNet', 'MaskNet'].")


def generate_ordered_input_vars(setup):
    inputs_list = list()
    for spcam_var in setup.spcam_inputs:
        if spcam_var.dimensions == 3:
            for level, _ in setup.parents_idx_levs:
                # There's enough info to build a Variable_Lev_Metadata list
                # However, it could be better to do a bigger reorganization
                var_name = f"{spcam_var.name}-{round(level, 2)}"
                inputs_list.append(Variable_Lev_Metadata.parse_var_name(var_name))
        elif spcam_var.dimensions == 2:
            var_name = spcam_var.name
            inputs_list.append(Variable_Lev_Metadata.parse_var_name(var_name))
    return inputs_list
