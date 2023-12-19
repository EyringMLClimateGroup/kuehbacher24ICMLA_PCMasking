# Implementation for CASTLE neural network
# Paper: Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery. https://doi.org/10/grw6pt
# Original code at https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/castle and
# https://github.com/trentkyono/CASTLE
import tensorflow as tf
from tensorflow import keras

from neural_networks.castle.castle_model_original import CASTLEOriginal
from neural_networks.castle.castle_model_adapted import CASTLEAdapted
from neural_networks.castle.castle_model_simplified import CASTLESimplified
from neural_networks.castle.legacy.castle_model import CASTLE


# Todo:
#  - Implement partial training
def build_castle(setup, num_x_inputs, learning_rate=0.001, eager_execution=False, strategy=None, seed=None):
    """
    Builds and compiles a neural network with CASTLE (Causal Structure Learning) regularization
    from Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery.
    https://doi.org/10/grw6pt.

    The output of the model is an array of shape [batch_size, num_x_inputs + 1].
    The first element of the output (output[:, 0]) contains the prediction for the target variable y, while
    the other outputs are reconstructions of the regressors x.

    Args:
        setup (utils.setup.SetupNeuralNetworks): A utils.setup.SetupNeuralNetworks instance containing
            specifics for CASTLE model creation (e.g. number of hidden layers, activation function, etc)
        num_x_inputs (int): The number of regressors, i.e. the x-variables.
        learning_rate (float): Optimizer learning rate: Defaults to 0.001.
        eager_execution (bool): If `True`, the code will be executed in eager mode and the model's logic will
            not be wrapped inside a tf.function. Can be used for debugging purposes. Defaults to `False`.
        strategy (tf.distribute.Strategy): State and compute distribution policy for parallelization
            across GPUs and/or SLURM nodes.
        seed (int): Random seed. Used to make the behavior of the initializer deterministic.
            Note that a seeded initializer will produce the same random values across multiple calls.

    Returns:
        tf.keras.Model: A tf.keras model designed with CASTLE regularization.

    Raises:
        ValueError: If the `setup.nn_type` is not one
                    `['CastleOriginal', 'CastleAdapted', 'CASTLESimplified', 'castleNN]`.
    """
    # Enable eager execution for debugging
    tf.config.run_functions_eagerly(eager_execution)
    # Force eager execution of tf.data functions as well
    if eager_execution:
        tf.data.experimental.enable_debug_mode()

    if setup.kernel_initializer_input_layers is None:
        setup.kernel_initializer_input_layers = {"initializer": "RandomNormal",
                                                 "mean": 0.0,
                                                 "std": 0.01}

    if setup.kernel_initializer_hidden_layers is None:
        setup.kernel_initializer_hidden_layers = {"initializer": "RandomNormal",
                                                  "mean": 0.0,
                                                  "std": 0.1}

    if setup.kernel_initializer_output_layers is None:
        setup.kernel_initializer_output_layers = {"initializer": "RandomNormal",
                                                  "mean": 0.0,
                                                  "std": 0.01}
    try:
        relu_alpha = setup.relu_alpha
    except AttributeError:
        relu_alpha = None

    def _build_castle():
        # Build model
        if setup.nn_type == "CASTLEOriginal":
            model_ = CASTLEOriginal(num_x_inputs, setup.hidden_layers, setup.activation, rho=setup.rho,
                                    alpha=setup.alpha,
                                    beta=setup.beta,
                                    lambda_weight=setup.lambda_weight,
                                    relu_alpha=relu_alpha,
                                    seed=seed,
                                    kernel_initializer_input_layers=setup.kernel_initializer_input_layers,
                                    kernel_initializer_hidden_layers=setup.kernel_initializer_hidden_layers,
                                    kernel_initializer_output_layers=setup.kernel_initializer_output_layers)
        elif setup.nn_type == "CASTLEAdapted":
            model_ = CASTLEAdapted(num_x_inputs, setup.hidden_layers, setup.activation, rho=setup.rho,
                                   alpha=setup.alpha,
                                   lambda_prediction=setup.lambda_prediction,
                                   lambda_sparsity=setup.lambda_sparsity,
                                   lambda_acyclicity=setup.lambda_acyclicity,
                                   lambda_reconstruction=setup.lambda_reconstruction,
                                   acyclicity_constraint=setup.acyclicity_constraint,
                                   relu_alpha=relu_alpha, seed=seed,
                                   kernel_initializer_input_layers=setup.kernel_initializer_input_layers,
                                   kernel_initializer_hidden_layers=setup.kernel_initializer_hidden_layers,
                                   kernel_initializer_output_layers=setup.kernel_initializer_output_layers)
        elif setup.nn_type == "CASTLESimplified":
            model_ = CASTLESimplified(num_x_inputs, setup.hidden_layers, setup.activation,
                                      lambda_sparsity=setup.lambda_sparsity,
                                      relu_alpha=relu_alpha, seed=seed,
                                      kernel_initializer_input_layers=setup.kernel_initializer_input_layers,
                                      kernel_initializer_hidden_layers=setup.kernel_initializer_hidden_layers,
                                      kernel_initializer_output_layers=setup.kernel_initializer_output_layers)
        elif setup.nn_type == "CastleNN":
            # Backwards compatibility for older CASTLE version
            model_ = CASTLE(num_x_inputs, setup.hidden_layers, setup.activation, rho=setup.rho, alpha=setup.alpha,
                            lambda_weight=setup.lambda_weight, relu_alpha=0.3, seed=seed)
        else:
            raise ValueError(f"Unknown CASTLE type {setup.nn_type}. "
                             f"Must be one of ['CastleOriginal', 'CastleAdapted'].")

        model_.build(input_shape=(None, num_x_inputs))
        # Compile model
        return _compile_castle(model_, learning_rate, eager_execution)

    if strategy is not None:
        with strategy.scope():
            model = _build_castle()
    else:
        model = _build_castle()
    return model


def _compile_castle(model, learning_rate, eager_execution):
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
