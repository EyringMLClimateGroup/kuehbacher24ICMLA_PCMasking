# Implementation for CASTLE neural network
# Paper: Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery. https://doi.org/10/grw6pt
# Original code at https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/castle and
# https://github.com/trentkyono/CASTLE
import tensorflow as tf
from tensorflow import keras

from neural_networks.castle.castle_model_original import CASTLEOriginal
from neural_networks.castle.castle_model_adapted import CASTLEAdapted


# Todo:
#  - Implement partial training
#  - Implement CASTLE code version of loss
def build_castle(setup, num_x_inputs, learning_rate=0.001, eager_execution=False, strategy=None, seed=None):
    """
    Builds and compiles a neural network with CASTLE (Causal Structure Learning) regularization
    from Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery.
    https://doi.org/10/grw6pt.

    The output of the model is an array of shape [num_x_inputs + 1, batch_size, 1].
    The first element of the output (output[0]) contains the prediction for the target variable y.

    Args:
        num_x_inputs (int): The number of predictors, i.e. the x-variables. This is also the number of neural network
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
        ValueError: If the CASTLE model flavor specified in setup is unknown.
            Known flavors are: 'CastleOriginal', 'CastleAdapted'
    """
    # Enable eager execution for debugging
    tf.config.run_functions_eagerly(eager_execution)
    # Force eager execution of tf.data functions as well
    if eager_execution:
        tf.data.experimental.enable_debug_mode()

    if setup.rho <= 0:
        raise ValueError("Penalty parameter `rho` for Lagrangian optimization scheme for acyclicity constraint "
                         "must be greater than 0.")

    kernel_initializer_input_layers = get_kernel_initializer(setup.kernel_initializer_input_layers, seed)
    if kernel_initializer_input_layers is None:
        kernel_initializer_input_layers = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=seed)

    kernel_initializer_hidden_layers = get_kernel_initializer(setup.kernel_initializer_hidden_layers,
                                                              seed)
    if kernel_initializer_hidden_layers is None:
        kernel_initializer_hidden_layers = keras.initializers.GlorotUniform(seed=seed)

    kernel_initializer_output_layers = get_kernel_initializer(setup.kernel_initializer_output_layers,
                                                              seed)
    if kernel_initializer_output_layers is None:
        kernel_initializer_output_layers = keras.initializers.HeUniform(seed=seed)

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
                                    kernel_initializer_input_layers=kernel_initializer_input_layers,
                                    kernel_initializer_hidden_layers=kernel_initializer_hidden_layers,
                                    kernel_initializer_output_layers=kernel_initializer_output_layers)
        elif setup.nn_type == "CASTLEAdapted":
            model_ = CASTLEAdapted(num_x_inputs, setup.hidden_layers, setup.activation, rho=setup.rho,
                                   alpha=setup.alpha,
                                   lambda_prediction=setup.lambda_prediction,
                                   lambda_sparsity=setup.lambda_sparsity,
                                   lambda_acyclicity=setup.lambda_acyclicity,
                                   lambda_reconstruction=setup.lambda_reconstruction,
                                   acyclicity_constraint=setup.acyclicity_constraint,
                                   relu_alpha=relu_alpha, seed=seed,
                                   kernel_initializer_input_layers=kernel_initializer_input_layers,
                                   kernel_initializer_hidden_layers=kernel_initializer_hidden_layers,
                                   kernel_initializer_output_layers=kernel_initializer_output_layers)
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


def get_kernel_initializer(kernel_initializer, seed):
    if kernel_initializer is None:
        kernel_initializer = None
    elif kernel_initializer["initializer"] == "Constant":
        kernel_initializer = keras.initializers.Constant(value=kernel_initializer["value"])
    elif kernel_initializer["initializer"] == "GlorotNormal":
        kernel_initializer = keras.initializers.GlorotNormal(seed=seed)
    elif kernel_initializer["initializer"] == "GlorotUniform":
        kernel_initializer = keras.initializers.GlorotUniform(seed=seed)
    elif kernel_initializer["initializer"] == "HeNormal":
        kernel_initializer = keras.initializers.HeNormal(seed=seed)
    elif kernel_initializer["initializer"] == "HeUniform":
        kernel_initializer = keras.initializers.HeUniform(seed=seed)
    elif kernel_initializer["initializer"] == "Identity":
        kernel_initializer = keras.initializers.Identity(gain=kernel_initializer["gain"])
    elif kernel_initializer["initializer"] == "LecunNormal":
        kernel_initializer = keras.initializers.LecunNormal(seed=seed)
    elif kernel_initializer["initializer"] == "LecunUniform":
        kernel_initializer = keras.initializers.LecunUniform(seed=seed)
    elif kernel_initializer["initializer"] == "Ones":
        kernel_initializer = keras.initializers.Ones()
    elif kernel_initializer["initializer"] == "Orthogonal":
        kernel_initializer = keras.initializers.Orthogonal(gain=kernel_initializer["gain"], seed=seed)
    elif kernel_initializer["initializer"] == "RandomNormal":
        kernel_initializer = keras.initializers.RandomNormal(mean=kernel_initializer["mean"],
                                                             stddev=kernel_initializer["std"], seed=seed)
    elif kernel_initializer["initializer"] == "RandomUniform":
        kernel_initializer = keras.initializers.RandomUniform(minval=kernel_initializer["min_val"],
                                                              maxval=kernel_initializer["max_val"],
                                                              seed=seed)
    elif kernel_initializer["initializer"] == "TruncatedNormal":
        kernel_initializer = keras.initializers.TruncatedNormal(mean=kernel_initializer["mean"],
                                                                stddev=kernel_initializer["std"], seed=seed)
    elif kernel_initializer["initializer"] == "VarianceScaling":
        kernel_initializer = keras.initializers.VarianceScaling(scale=kernel_initializer["scale"],
                                                                mode=kernel_initializer["mode"],
                                                                distribution=kernel_initializer["distribution"],
                                                                seed=seed)
    elif kernel_initializer["initializer"] == "Zeros":
        kernel_initializer = keras.initializers.Zeros()
    else:
        raise ValueError(f"Unknown value for kernel initializer: {kernel_initializer}. Possible values are "
                         f"['Constant', 'GlorotNormal', 'GlorotUniform', 'HeNormal', 'HeUniform', 'Identity', "
                         f"'LecunNormal', 'LecunUniform', 'Ones', 'Orthogonal', 'RandomNormal', 'RandomUniform', "
                         f"'TruncatedNormal', 'VarianceScaling', 'Zeros'].")

    return kernel_initializer
