# Implementation for CASTLE neural network
# Paper: Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery. https://doi.org/10/grw6pt
# Original code at https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/castle and
# https://github.com/trentkyono/CASTLE
import tensorflow as tf
from tensorflow import keras
from neural_networks.castle.castle_model import CASTLE, mse_x


# Todo:
#  - Implement partial training
#  - Implement CASTLE code version of loss
def build_castle(num_inputs, hidden_layers, activation, rho, alpha, lambda_weight, learning_rate=0.001,
                 eager_execution=False, strategy=None, seed=None):
    """
    Implement neural network with CASTLE (Causal Structure Learning) regularization
    from Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery.
    https://doi.org/10/grw6pt.

    The output of the model is an array of shape [num_inputs + 1, batch_size, 1].
    The first element of the output (output[0]) contains the prediction for the target variable y.

    Args:
        num_inputs (int): The number of predictors, i.e. the x-variables. This is also the number of neural network
            inputs for all input sub-layers.
        hidden_layers (list of int): A list containing the hidden units for all hidden layers.
            ``len(hidden_layers)`` gives the number of hidden layers.
        activation (str, case insensitive): A string specifying the activation function,
            e.g. `relu`, `linear`, `sigmoid`, `tanh`. In addition to tf.keras specific strings for
            built-in activation functions, `LeakyReLU` can be used to specify leaky ReLU activation function.
            See also https://www.tensorflow.org/api_docs/python/tf/keras/layers/Activation.
        rho (float): Penalty parameter for Lagrangian optimization scheme for acyclicity constraint.
            `rho` must be greater than 0.
        alpha (float): Lagrangian multiplier for Lagrangian optimization scheme for acyclicity constraint.
        lambda_weight (float): Weighting coefficient for the regularization term in the training loss.
        learning_rate (float): Optimizer learning rate: Defaults to 0.001.
        eager_execution (bool): If `True`, the code will be executed eagerly and the model's logic will
            not be wrapped inside a tf.function. Can be used for debugging purposes. Defaults to `False`.
        strategy (tf.distribute.Strategy): State and compute distribution policy for parallelization
            across GPUs and/or SLURM nodes.
        seed (int): Random seed. Used to make the behavior of the initializer deterministic.
            Note that a seeded initializer will produce the same random values across multiple calls.

    Returns:
        tf.keras.Model: A tf.keras model designed according to CASTLE architecture.

    Raises:
        ValueError: If `rho` is not greater than 0.
    """
    print("Using custom CASTLE model with compute_loss. ")

    # Enable eager execution for debugging
    tf.config.run_functions_eagerly(eager_execution)
    # Force eager execution of tf.data functions as well
    if eager_execution:
        tf.data.experimental.enable_debug_mode()

    if rho <= 0:
        raise ValueError("Penalty parameter `rho` for Lagrangian optimization scheme for acyclicity constraint "
                         "must be greater than 0.")

    def _build_castle():
        # Build model
        model_ = CASTLE(num_inputs, hidden_layers, activation, rho=rho, alpha=alpha, lambda_weight=lambda_weight,
                        relu_alpha=0.3, seed=seed)
        model_.build(input_shape=(None, num_inputs))
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
        metrics=[mse_x],
        run_eagerly=eager_execution
    )

    return model
