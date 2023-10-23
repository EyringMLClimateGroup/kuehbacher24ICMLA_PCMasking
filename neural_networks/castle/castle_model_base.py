# Implementation for CASTLE neural network
# Paper: Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery. https://doi.org/10/grw6pt
# Original code at https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/castle and
# https://github.com/trentkyono/CASTLE
import tensorflow as tf
from tensorflow import keras

from neural_networks.castle.masked_dense_layer import MaskedDenseLayer


@tf.keras.utils.register_keras_serializable()
class CASTLEBase(keras.Model):
    """A neural network model with CASTLE (Causal Structure Learning) regularization adapted
    from Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery.
    https://doi.org/10/grw6pt.

    The output of the model is an array of shape [num_inputs + 1, batch_size, 1].
    The first element of the output (output[0]) contains the prediction for the target variable y.

    Key differences to the original paper:
      - Target y is not passed as an input into the network.
      - Sparsity loss uses the matrix L1-norm and is averaged over the number of input layers.

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
        relu_alpha (float): Negative float coefficient for leaky ReLU activation function. Default: 0.3.
        lambda_weight (float): Weighting coefficient for the regularization term in the training loss.
        seed (int): Random seed. Used to make the behavior of the initializer deterministic.
            Note that a seeded initializer will produce the same random values across multiple calls.
        name (str): Name of the model. Default: "castle_model".
        **kwargs: Keyword arguments.
    """

    def __init__(self, num_inputs, hidden_layers, activation, rho, alpha, relu_alpha=0.3, seed=None,
                 name="castle_model", **kwargs):
        super(CASTLEBase, self).__init__(name=name, **kwargs)

        # Set attributes
        self.rho = rho
        self.alpha = alpha
        self.seed = seed

        self.activation = activation.lower()
        self.relu_alpha = relu_alpha

        self.num_outputs = 1
        self.num_inputs = num_inputs

        self.hidden_layers = hidden_layers

        # The following attributes are CASTLE flavor specific and need to be set when subclassing CASTLEBase
        self.num_input_layers = None

        self.input_sub_layers = None
        self.shared_hidden_layers = None
        self.output_sub_layers = None

        # Metrics
        self.metric_dict = dict()
        self.metric_dict["loss_tracker"] = keras.metrics.Mean(name="loss")
        self.metric_dict["prediction_loss_tracker"] = keras.metrics.Mean(name="prediction_loss")
        self.metric_dict["reconstruction_loss_tracker"] = keras.metrics.Mean(name="reconstruction_loss")
        self.metric_dict["sparsity_loss_tracker"] = keras.metrics.Mean(name="sparsity_loss")
        self.metric_dict["acyclicity_loss_tracker"] = keras.metrics.Mean(name="acyclicity_loss")
        self.metric_dict["mse_x_tracker"] = keras.metrics.Mean(name="mse_x")
        self.metric_dict["mse_y_tracker"] = keras.metrics.Mean(name="mse_y")

        # Masks: This is instantiated here but set in
        self.masks = list()

    def reset_metrics(self):
        """Resets the state of all the metrics in the model."""
        for metric in self.metric_dict.values():
            metric.reset()

    @property
    def metrics(self):
        """Returns a list of model metrics."""
        # We list our `Metric` objects here so that `reset_state()` can be
        # called automatically at the start of each epoch or at the start of `evaluate()`..
        return list(self.metric_list.values())

    @staticmethod
    def compute_prediction_loss(y_true, yx_pred):
        """Computes CASTLE prediction loss."""
        return tf.reduce_mean(keras.losses.mse(y_true, yx_pred[:, 0]), name="prediction_loss_reduce_mean")

    @staticmethod
    def compute_reconstruction_loss_x(x_true, yx_pred):
        """Computes CASTLE reconstruction loss."""
        # Frobenius norm between all inputs and outputs averaged over the number of samples in the batch
        return tf.reduce_mean(tf.norm(tf.subtract(tf.expand_dims(x_true, axis=-1), yx_pred[:, 1:]),
                                      ord='fro', axis=[-2, -1]),
                              name="reconstruction_loss_reduce_mean")

    @staticmethod
    def compute_reconstruction_loss_yx(yx_true, yx_pred):
        """Computes CASTLE reconstruction loss."""
        # Frobenius norm between all inputs and outputs averaged over the number of samples in the batch
        return tf.reduce_mean(tf.norm(tf.subtract(tf.expand_dims(yx_true, axis=-1), yx_pred),
                                      ord='fro', axis=[-2, -1]),
                              name="reconstruction_loss_reduce_mean")

    def compute_acyclicity_loss(self, input_layer_weights, acyclicity_loss_func):
        """Computes the values of the NOTEARS acyclicity constraint."""
        # Compute matrix with l2 - norms of input sub-layer weight matrices:
        # The entry [l2_norm_matrix]_(k,j) is the l2-norm of the k-th row of the weight matrix in input sub-layer j.
        # Since our weight matrices are of dimension dxd (d is the number of x-variables), but we have d+1
        # variables all together (x-variables and y) we set the first row 0 for y.
        l2_norm_matrix = list()
        for j, w in enumerate(input_layer_weights):
            l2_norm_matrix.append(tf.concat([tf.zeros((1,), dtype=tf.float32),
                                             tf.norm(w, axis=1, ord=2, name="l2_norm_input_layers")], axis=0))
        l2_norm_matrix = tf.stack(l2_norm_matrix, axis=1)

        h = acyclicity_loss_func(l2_norm_matrix)
        # tf.print(f"h function = {h}")

        # Acyclicity loss is computed using the Lagrangian scheme with penalty parameter rho and
        # Lagrangian multiplier alpha.
        h_squared = tf.math.square(h, name="h_squared")
        return tf.math.add(tf.math.multiply(tf.math.multiply(0.5, self.rho), h_squared, name="lagrangian_penalty"),
                           tf.math.multiply(self.alpha, h, name="lagrangian_optimizer"),
                           name="acyclicity_loss")

    def compute_sparsity_loss(self, input_layer_weights):
        """Computes sparsity loss as the sum of the matrix L1 norm of the input layer weight matrices."""
        # Compute the matrix l1 - norm (maximum absolute column sum norm) for the weight matrices in the input_sublayer
        sparsity_regularizer = 0.0
        sparsity_regularizer += tf.reduce_sum(
            tf.norm(input_layer_weights[0], ord=1, axis=[-2, -1], name="l1_norm_input_layers"))
        for i, weight in enumerate(input_layer_weights[1:]):
            # Ignore the masked row
            w_1 = tf.slice(weight, [0, 0], [i, -1])
            w_2 = tf.slice(weight, [i + 1, 0], [-1, -1])

            sparsity_regularizer += tf.norm(w_1, ord=1, axis=[-2, -1], name="l1_norm_input_layers") \
                                    + tf.norm(w_2, ord=1, axis=[-2, -1], name="l1_norm_input_layers")

        # Scale with number of input layers
        sparsity_regularizer = sparsity_regularizer / len(input_layer_weights)
        return sparsity_regularizer

    @staticmethod
    def compute_mse_x(x_true, yx_pred):
        """Computes the MSE between inputs x values and the predicted reconstructions."""
        return tf.metrics.mse(tf.expand_dims(x_true, axis=-1), yx_pred[:, 1:])

    @staticmethod
    def compute_mse_y(y_true, yx_pred):
        """Computes the MSE between the actual label y and its prediction."""
        return tf.metrics.mse(y_true, yx_pred[:, 0])


def build_graph(num_input_layers, num_inputs, num_outputs, hidden_layers, relu_alpha, activation, seed):
    """Initializes the network layers."""
    # Create layers
    # Get activation function
    act_func = tf.keras.layers.LeakyReLU(alpha=relu_alpha) if activation == "leakyrelu" \
        else tf.keras.layers.Activation(activation)

    # Input sub-layers: One sub-layer for each input and each sub-layers receives all the inputs
    # We're using RandomNormal initializers for kernel and bias because the original CASTLE
    # implementation used random normal initialization.
    input_sub_layers = list()
    # First input layer is not masked
    dense_layer = keras.layers.Dense(hidden_layers[0], activation=act_func, name=f"input_sub_layer_0",
                                     kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01,
                                                                                        seed=seed))
    input_sub_layers.append(dense_layer)

    for i in range(num_input_layers - 1):
        mask = tf.transpose(
            tf.one_hot([i] * hidden_layers[0], depth=num_inputs, on_value=0.0, off_value=1.0, axis=-1))
        masked_dense_layer = MaskedDenseLayer(hidden_layers[0], mask, activation=act_func,
                                              name=f"input_sub_layer_{i + 1}",
                                              kernel_initializer=keras.initializers.RandomNormal(mean=0.0,
                                                                                                 stddev=0.01,
                                                                                                 seed=seed))
        input_sub_layers.append(masked_dense_layer)

    # Shared hidden layers: len(hidden_layers) number of hidden layers. All input sub-layers feed into the
    #   same (shared) hidden layers.
    shared_hidden_layers = list()
    for i, n_hidden_layer_nodes in enumerate(hidden_layers):
        shared_hidden_layers.append(
            keras.layers.Dense(n_hidden_layer_nodes, activation=act_func, name=f"shared_hidden_layer_{i}",
                               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1,
                                                                                  seed=seed)))

    # Output sub-layers: One sub-layer for each input. Each output layer outputs one value, i.e.
    #   reconstructs one input.
    output_sub_layers = list()
    for i in range(num_input_layers):
        output_sub_layers.append(
            # No activation function, i.e. linear
            keras.layers.Dense(num_outputs, activation="linear", name=f"output_sub_layer_{i}",
                               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01,
                                                                                  seed=seed)))
    return input_sub_layers, shared_hidden_layers, output_sub_layers


def compute_h(matrix, castle_computation=True):
    """Compute the acyclicity constraint function for a (d x d)-matrix M::

        h(M) = tr(e^(M * M)) - d

    where `*` denotes the Hadamard (element-wise) product.

    See Zheng rt sl. 2018. DAGs with NO TEARS: Continuous Optimization for Structure Learning. https://doi.org/10/grxdgr
    and Zheng et al. 2019. Learning Sparse Nonparametric DAGs. https://doi.org/10/grxsr9
    for details.

    Args:
        matrix (tensor or numpy array): A matrix with shape with shape `[..., d, d]`.
        castle_computation (bool): If `True`, uses the same truncated power series as in original implementation
            from Kyono et al. 2020 to compute the matrix exponential. Otherwise, the Tensorflow function
            `tf.linalg.expm` is used for matrix exponential computation.

    Returns:
        float tensor: Value of the acyclicity constraint function.
    """
    d = matrix.shape[0]

    if castle_computation:
        # Truncated power series from https://github.com/trentkyono/CASTLE
        coff = 1.0

        z = tf.math.multiply(matrix, matrix)
        dag_l = tf.cast(d, tf.float32)

        z_in = tf.eye(d)
        for i in range(1, 10):
            z_in = tf.matmul(z_in, z)

            dag_l += 1. / coff * tf.linalg.trace(z_in)
            coff = coff * (i + 1)

        # tf.print(f"dag loss = {dag_l}")
        return dag_l - tf.cast(d, dtype=tf.float32)

    # Else: Compute using tf.linalg.expm
    hadamard_product = tf.math.multiply(matrix, matrix)
    # tf.print(f"Hadamard product = {hadamard_product}")
    matrix_exp = tf.linalg.expm(hadamard_product)
    # tf.print(f"matrix exponential = {matrix_exp}")
    return tf.linalg.trace(matrix_exp) - tf.cast(d, dtype=tf.float32)
