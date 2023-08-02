# Implementation for CASTLE neural network
# Paper: Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery. https://doi.org/10/grw6pt
# Original code at https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/castle and
# https://github.com/trentkyono/CASTLE
import tensorflow as tf
from tensorflow import keras


@tf.keras.utils.register_keras_serializable()
class CASTLE(keras.Model):
    def __init__(self, num_inputs, hidden_layers, activation, rho, alpha, reg_lambda, relu_alpha=0.3, seed=None):
        super().__init__()
        self.rho = rho
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.seed = seed

        self.activation = activation.lower()
        self.relu_alpha = relu_alpha

        self.num_outputs = 1
        self.num_inputs = num_inputs
        # The number of input sub-layers is nn_inputs + 1, because we need one sub-layer with all
        # x-variables for the prediction of y, and num_inputs layers for the prediction for each of x-variable
        self.num_input_layers = self.num_inputs + 1
        self.hidden_layers = hidden_layers

        # Names of the inputs layers. The name has to be the same as in self._build_graph
        self.input_layer_weight_names = [f"input_sub_layer_{i}/kernel:0" for i in range(self.num_input_layers)]

        # Metrics
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.prediction_loss_tracker = keras.metrics.Mean(name="prediction_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.sparsity_loss_tracker = keras.metrics.Mean(name="sparsity_loss")
        self.acyclicity_loss_tracker = keras.metrics.Mean(name="acyclicity_loss")

        self._build_graph()

    def _build_graph(self):
        # Create layers
        # Get activation function
        act_func = tf.keras.layers.LeakyReLU(alpha=self.relu_alpha) if self.activation == "leakyrelu" \
            else tf.keras.layers.Activation(self.activation)

        # Input sub-layers: One sub-layer for each input and each sub-layers receives all the inputs
        # We're using RandomNormal initializers for kernel and bias because the original CASTLE
        # implementation used random normal initialization.
        # Todo: Check the standard deviation for the input layers. In CASTLE it's initialized as
        #  tf.Variable(tf.random_normal([self.n_hidden], seed=seed) * 0.01)
        #  and default values are mean=0.0, stddev=1.0
        self.input_sub_layers = list()
        for i in range(self.num_input_layers):
            self.input_sub_layers.append(
                keras.layers.Dense(self.hidden_layers[0], activation=act_func, name=f"input_sub_layer_{i}",
                                   kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01,
                                                                                      seed=self.seed),
                                   bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01,
                                                                                    seed=self.seed)))

        # Shared hidden layers: len(hidden_layers) number of hidden layers. All input sub-layers feed into the
        #   same (shared) hidden layers.
        self.shared_hidden_layers = list()
        for i, n_hidden_layer_nodes in enumerate(self.hidden_layers):
            self.shared_hidden_layers.append(
                keras.layers.Dense(n_hidden_layer_nodes, activation=act_func, name=f"shared_hidden_layer_{i}",
                                   kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1,
                                                                                      seed=self.seed),
                                   bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1,
                                                                                    seed=self.seed)))

        # Output sub-layers: One sub-layer for each input. Each output layer outputs one value, i.e.
        #   reconstructs one input.
        self.output_sub_layers = list()
        for i in range(self.num_input_layers):
            self.output_sub_layers.append(
                # No activation function, i.e. linear
                keras.layers.Dense(self.num_outputs, activation="linear", name=f"output_sub_layer_{i}",
                                   kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1,
                                                                                      seed=self.seed),
                                   bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1,
                                                                                    seed=self.seed)))

    def build(self, input_shape):
        super().build(input_shape)

        # Mask matrix rows in input layers so that a variable cannot cause itself
        #    Since the first input sub-layer aims to use all the x-variables to predict y, it does not need a mask.
        #    For the other sub-layers, we need to mask the x-variable that is to be predicted.
        # This can only be done after the graph is created because the layer weights are initialized only then
        for i in range(self.num_input_layers - 1):
            weights, bias = self.input_sub_layers[i + 1].get_weights()
            # Create one hot matrix that masks the row of the current input
            mask = tf.transpose(
                tf.one_hot([i] * self.hidden_layers[0], depth=self.num_inputs, on_value=0.0, off_value=1.0, axis=-1))
            self.input_sub_layers[i + 1].set_weights([weights * mask, bias])

    def call(self, inputs, **kwargs):
        # Create network graph
        inputs_hidden = [in_sub_layer(inputs) for in_sub_layer in self.input_sub_layers]

        hidden_outputs = list()
        for hidden_layer in self.shared_hidden_layers:
            # Pass all inputs through same hidden layers
            for x in inputs_hidden:
                hidden_outputs.append(hidden_layer(x))

            # Outputs become new inputs for next hidden layers
            inputs_hidden = hidden_outputs[-self.num_input_layers:]

        yx_outputs = [out_layer(x) for x, out_layer in
                      zip(hidden_outputs[-self.num_input_layers:], self.output_sub_layers)]
        # Concatenate the outputs into one tensor
        return tf.concat(yx_outputs, axis=1)

    # If we are using distribute strategy, we cannot decorate this with tf.function
    # @tf.function
    def train_step(self, data):
        # Unpack data
        x, y = data

        with tf.GradientTape() as tape:
            yx_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            acyclicity_loss, loss, prediction_loss, reconstruction_loss, sparsity_regularizer = self.loss_fn(x, y,
                                                                                                             yx_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.prediction_loss_tracker.update_state(prediction_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.sparsity_loss_tracker.update_state(sparsity_regularizer)
        self.acyclicity_loss_tracker.update_state(acyclicity_loss)

        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result(), "prediction_loss": self.prediction_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "sparsity_loss": self.sparsity_loss_tracker.result(),
                "acyclicity_loss": self.acyclicity_loss_tracker.result()}

    # If we are using distribute strategy, we cannot decorate this with tf.function
    # @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        yx_pred = self(x, training=False)

        # Compute loss
        acyclicity_loss, loss, prediction_loss, reconstruction_loss, sparsity_regularizer = self.loss_fn(x, y, yx_pred)

        # Update the metrics
        self.loss_tracker.update_state(loss)
        self.prediction_loss_tracker.update_state(prediction_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.sparsity_loss_tracker.update_state(sparsity_regularizer)
        self.acyclicity_loss_tracker.update_state(acyclicity_loss)

        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result(), "prediction_loss": self.prediction_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "sparsity_loss": self.sparsity_loss_tracker.result(),
                "acyclicity_loss": self.acyclicity_loss_tracker.result()}

    def loss_fn(self, x, y, yx_pred):
        reconstruction_loss = self.compute_reconstruction_loss(x, yx_pred)
        acyclicity_loss = self.compute_acyclicity_loss()
        sparsity_regularizer = self.compute_sparsity_loss()
        regularization_loss = reconstruction_loss + acyclicity_loss + sparsity_regularizer
        prediction_loss = self.compute_prediction_loss(y, yx_pred)
        loss = prediction_loss + self.reg_lambda * regularization_loss
        return acyclicity_loss, loss, prediction_loss, reconstruction_loss, sparsity_regularizer

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch or at the start of `evaluate()`..
        return [self.loss_tracker, self.prediction_loss_tracker, self.reconstruction_loss_tracker,
                self.sparsity_loss_tracker, self.acyclicity_loss_tracker]

    @staticmethod
    def compute_prediction_loss(y_true, yx_pred):
        return tf.reduce_mean(keras.losses.mse(y_true, yx_pred[:, 0]), name="prediction_loss_reduce_mean")

    @staticmethod
    def compute_reconstruction_loss(x_true, yx_pred):
        # Frobenius norm between all inputs and outputs averaged over the number of samples in the batch
        return tf.reduce_mean(tf.norm(x_true - yx_pred[:, 1:], ord='fro', axis=[-2, -1]),
                              name="reconstruction_loss_reduce_mean")

    def compute_acyclicity_loss(self):
        # Compute matrix with l2 - norms of input sub-layer weight matrices:
        # The entry [l2_norm_matrix]_(k,j) is the l2-norm of the k-th row of the weight matrix in input sub-layer j.
        # Since our weight matrices are of dimension dxd (d is the number of x-variables), but we have d+1
        # variables all together (x-variables and y) we set the first row 0 for y.
        input_layer_weights = [w for w in self.trainable_variables if w.name in self.input_layer_weight_names]

        l2_norm_matrix = list()
        for j, w in enumerate(input_layer_weights):
            l2_norm_matrix.append(tf.concat([tf.zeros((1,), dtype=tf.float32),
                                             tf.norm(w, axis=1, ord=2, name="l2_norm_input_layers")], axis=0))
        l2_norm_matrix = tf.stack(l2_norm_matrix, axis=1)

        h = compute_h(l2_norm_matrix)
        # tf.print(f"h function = {h}")

        # Acyclicity loss is computed using the Lagrangian scheme with penalty parameter rho and
        # Lagrangian multiplier alpha.
        h_squared = tf.math.square(h, name="h_squared")
        return tf.math.add(tf.math.multiply(0.5 * self.rho, h_squared, name="lagrangian_penalty"),
                           tf.math.multiply(self.alpha, h, name="lagrangian_optimizer"),
                           name="acyclicity_loss")

    def compute_sparsity_loss(self):
        # Compute the l1 - norm for the weight matrices in the input_sublayer
        input_layer_weights = [w for w in self.trainable_variables if w.name in self.input_layer_weight_names]

        sparsity_regularizer = 0.0
        sparsity_regularizer += tf.reduce_sum(tf.norm(input_layer_weights[0], ord=1, axis=1))
        for i, weight in enumerate(input_layer_weights[1:]):
            # Ignore the masked row
            w_1 = tf.slice(weight, [0, 0], [i - 1, -1])
            w_2 = tf.slice(weight, [i, 0], [-1, -1])

            sparsity_regularizer += tf.reduce_sum(tf.norm(w_1, ord=1, axis=1, name="l1_norm_input_layers"),
                                                  name="w1_reduce_sum") + \
                                    tf.reduce_sum(tf.norm(w_2, ord=1, axis=1, name="l1_norm_input_layers"),
                                                  name="w2_reduce_sum")
        return sparsity_regularizer

    def get_config(self):
        config = super(CASTLE, self).get_config()
        # These are the constructor arguments
        config.update(
            {
                "num_inputs": self.num_inputs,
                "hidden_layers": self.hidden_layers,
                "activation": self.activation,
                "rho": self.rho,
                "alpha": self.alpha,
                "reg_lambda": self.reg_lambda,
                "relu_alpha": self.relu_alpha,
                "seed": self.seed,
            }
        )
        return config


def compute_h(matrix, castle_computation=True):
    """
    Compute the acyclicity constraint function for a (d x d)-matrix M::

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


def mse_x(x_true, yx_pred):
    return tf.metrics.mse(x_true, yx_pred[:, 1:])
