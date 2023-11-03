# Implementation for CASTLE neural network
# Paper: Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery. https://doi.org/10/grw6pt
# Original code at https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/castle and
# https://github.com/trentkyono/CASTLE
import tensorflow as tf

from neural_networks.castle.castle_model_base import CASTLEBase, build_graph, compute_h_matrix_exp


@tf.keras.utils.register_keras_serializable()
class CASTLEOriginal(CASTLEBase):
    """A neural network model with CASTLE (Causal Structure Learning) regularization adapted
    from Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery.
    https://doi.org/10/grw6pt.

    The output of the model is an array of shape `[batch_size, num_x_inputs + 1]`.
    The first element of the output (`output[:, 0]`) contains the prediction for the target variable `y`, while
    the other outputs are reconstructions of the regressors `x`.

    As in the original paper, the model receives inputs of the shape `[y, x_1, ..., x_d]`,
    where `y` is the label to be predicted and x_i are the regressors. There is one input sub-layer for
    all elements in the input vector (i.e. there are `num_x_inputs` + 1 input sub-layers).
    The computation of the sparsity loss is slightly adapted from the paper, as in that the matrix
    L1-norm is used for its computation and the sparsity loss is averaged over the number of input layers.

    Args:
        num_x_inputs (int): The number of regressors, i.e. the x-variables.
        hidden_layers (list of int): A list containing the hidden units for all hidden layers.
            ``len(hidden_layers)`` gives the number of hidden layers.
        activation (str, case insensitive): A string specifying the activation function,
            e.g. `relu`, `linear`, `sigmoid`, `tanh`. In addition to tf.keras specific strings for
            built-in activation functions, `LeakyReLU` can be used to specify leaky ReLU activation function.
            See also https://www.tensorflow.org/api_docs/python/tf/keras/layers/Activation.
        rho (float): Penalty parameter for Lagrangian optimization scheme for acyclicity constraint.
            `rho` must be greater than 0.
        alpha (float): Lagrangian multiplier for Lagrangian optimization scheme for acyclicity constraint.
        beta (float): Weighting coefficient for sparsity loss.
        lambda_weight (float): Weighting coefficient for sum of all regularization losses.
        relu_alpha (float): Negative slope coefficient for leaky ReLU activation function. Default: 0.3.
        seed (int): Random seed. Used to make the behavior of the initializer deterministic.
            Note that a seeded initializer will produce the same random values across multiple calls.
        kernel_initializer_input_layers (tf.keras.initializers.Initializer): Initializer for the kernel
            weights matrix of the dense input layer.
        kernel_initializer_hidden_layers (tf.keras.initializers.Initializer): Initializer for the kernel
            weights matrix of the dense hidden layer.
        kernel_initializer_output_layers (tf.keras.initializers.Initializer): Initializer for the kernel
            weights matrix of the dense output layer.
        bias_initializer_input_layers (tf.keras.initializers.Initializer): Initializer for the bias vector
            of the dense input layer.
        bias_initializer_hidden_layers (tf.keras.initializers.Initializer): Initializer for the bias vector
            of the dense hidden layer.
        bias_initializer_output_layers (tf.keras.initializers.Initializer): Initializer for the bias vector
            of the dense output layer.
        kernel_regularizer_input_layers (tf.keras.regularizers.Regularizer): Regularizer function applied
            to the kernel weights matrix of the dense input layer.
        kernel_regularizer_hidden_layers (tf.keras.regularizers.Regularizer): Regularizer function applied
            to the kernel weights matrix of the dense hidden layer.
        kernel_regularizer_output_layers (tf.keras.regularizers.Regularizer): Regularizer function applied
            to the kernel weights matrix of the dense output layer.
        bias_regularizer_input_layers (tf.keras.regularizers.Regularizer): Regularizer function applied
            to the bias vector of the dense input layer.
        bias_regularizer_hidden_layers (tf.keras.regularizers.Regularizer): Regularizer function applied
            to the bias vector of the dense hidden layer.
        bias_regularizer_output_layers (tf.keras.regularizers.Regularizer): Regularizer function applied
            to the bias vector of the dense output layer.
        activity_regularizer_input_layers (tf.keras.regularizers.Regularizer): Regularizer function applied
             to the output of the dense input layer (its "activation").
        activity_regularizer_hidden_layers (tf.keras.regularizers.Regularizer): Regularizer function applied
             to the output of the dense hidden layer (its "activation").
        activity_regularizer_output_layers (tf.keras.regularizers.Regularizer): Regularizer function applied
             to the output of the dense output layer (its "activation").
        name (string) : Name of the model. Default: "castle_original".
        **kwargs: Keyword arguments.
    """

    def __init__(self,
                 num_x_inputs,
                 hidden_layers,
                 activation,
                 rho,
                 alpha,
                 beta,
                 lambda_weight,
                 relu_alpha=0.3,
                 seed=None,
                 kernel_initializer_input_layers=None,
                 kernel_initializer_hidden_layers=None,
                 kernel_initializer_output_layers=None,
                 bias_initializer_input_layers=None,
                 bias_initializer_hidden_layers=None,
                 bias_initializer_output_layers=None,
                 kernel_regularizer_input_layers=None,
                 kernel_regularizer_hidden_layers=None,
                 kernel_regularizer_output_layers=None,
                 bias_regularizer_input_layers=None,
                 bias_regularizer_hidden_layers=None,
                 bias_regularizer_output_layers=None,
                 activity_regularizer_input_layers=None,
                 activity_regularizer_hidden_layers=None,
                 activity_regularizer_output_layers=None,
                 name="castle_original", **kwargs):
        num_input_layers = num_x_inputs + 1
        num_outputs = 1

        # Initialize the model as a functional (custom) model
        # Original CASTLE also passes Y as an input to the network. Use `with_y=True`
        input_sub_layers, shared_hidden_layers, output_sub_layers = build_graph(num_input_layers=num_input_layers,
                                                                                num_x_inputs=num_x_inputs,
                                                                                num_outputs=num_outputs,
                                                                                hidden_layers=hidden_layers,
                                                                                activation=activation.lower(),
                                                                                kernel_initializer_input_layers=kernel_initializer_input_layers,
                                                                                kernel_initializer_hidden_layers=kernel_initializer_hidden_layers,
                                                                                kernel_initializer_output_layers=kernel_initializer_output_layers,
                                                                                bias_initializer_input_layers=bias_initializer_input_layers,
                                                                                bias_initializer_hidden_layers=bias_initializer_hidden_layers,
                                                                                bias_initializer_output_layers=bias_initializer_output_layers,
                                                                                kernel_regularizer_input_layers=kernel_regularizer_input_layers,
                                                                                kernel_regularizer_hidden_layers=kernel_regularizer_hidden_layers,
                                                                                kernel_regularizer_output_layers=kernel_regularizer_output_layers,
                                                                                bias_regularizer_input_layers=bias_regularizer_input_layers,
                                                                                bias_regularizer_hidden_layers=bias_regularizer_hidden_layers,
                                                                                bias_regularizer_output_layers=bias_regularizer_output_layers,
                                                                                activity_regularizer_input_layers=activity_regularizer_input_layers,
                                                                                activity_regularizer_hidden_layers=activity_regularizer_hidden_layers,
                                                                                activity_regularizer_output_layers=activity_regularizer_output_layers,
                                                                                relu_alpha=relu_alpha,
                                                                                seed=seed, with_y=True)

        inputs = tf.keras.Input(shape=(num_x_inputs + 1,))
        # Create network graph
        inputs_hidden = [in_sub_layer(inputs) for in_sub_layer in input_sub_layers]

        hidden_outputs = list()
        for hidden_layer in shared_hidden_layers:
            # Pass all inputs through same hidden layers
            for x in inputs_hidden:
                hidden_outputs.append(hidden_layer(x))

            # Outputs become new inputs for next hidden layers
            inputs_hidden = hidden_outputs[-num_input_layers:]

        yx_outputs = [out_layer(x) for x, out_layer in
                      zip(hidden_outputs[-num_input_layers:], output_sub_layers)]

        # Stack outputs into one tensor
        # outputs = tf.squeeze(tf.stack(yx_outputs, axis=1))
        outputs = tf.stack(yx_outputs, axis=1, name="stack_outputs")
        outputs = tf.reshape(outputs, shape=(tf.shape(outputs)[0], tf.shape(outputs)[1]), name="reshape_output_dims")

        super(CASTLEOriginal, self).__init__(num_x_inputs=num_x_inputs, hidden_layers=hidden_layers,
                                             activation=activation, rho=rho, alpha=alpha, seed=seed,
                                             kernel_initializer_input_layers=kernel_initializer_input_layers,
                                             kernel_initializer_hidden_layers=kernel_initializer_hidden_layers,
                                             kernel_initializer_output_layers=kernel_initializer_output_layers,
                                             bias_initializer_input_layers=bias_initializer_input_layers,
                                             bias_initializer_hidden_layers=bias_initializer_hidden_layers,
                                             bias_initializer_output_layers=bias_initializer_output_layers,
                                             kernel_regularizer_input_layers=kernel_regularizer_input_layers,
                                             kernel_regularizer_hidden_layers=kernel_regularizer_hidden_layers,
                                             kernel_regularizer_output_layers=kernel_regularizer_output_layers,
                                             bias_regularizer_input_layers=bias_regularizer_input_layers,
                                             bias_regularizer_hidden_layers=bias_regularizer_hidden_layers,
                                             bias_regularizer_output_layers=bias_regularizer_output_layers,
                                             activity_regularizer_input_layers=activity_regularizer_input_layers,
                                             activity_regularizer_hidden_layers=activity_regularizer_hidden_layers,
                                             activity_regularizer_output_layers=activity_regularizer_output_layers,
                                             name=name, inputs=inputs, outputs=outputs, **kwargs)

        self.num_input_layers = num_input_layers
        self.lambda_weight = lambda_weight

        self.beta = beta
        self.relu_alpha = relu_alpha

        self.input_sub_layers = input_sub_layers
        self.shared_hidden_layers = shared_hidden_layers
        self.output_sub_layers = output_sub_layers

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        """
        Compute model loss. Overrides base method.

        Args:
            x: Input data. Here, this is actually [y, x].
            y: Target data. This is `None`, as the target data is already part of the input data.
            y_pred: Predictions returned by the model (output of `model(x)`). In the case of CASTLE,
              `y_pred` contains the prediction for `y` and the reconstruction of `x`.
            sample_weight: Sample weights for weighting the loss function.

        Returns:
            The total loss as a `tf.Tensor`, or `None` if no loss results (which
            is the case when called by `Model.test_step`).
        """
        # In this case, x contains the true y and x: [y, x]
        y = x[:, 0]

        input_layer_weights = [layer.trainable_variables[0] * layer.mask for layer in self.input_sub_layers]

        # In CASTLE, y_pred is (y_pred, x_pred)
        prediction_loss = self.compute_prediction_loss(y, y_pred)

        reconstruction_loss = self.compute_reconstruction_loss_yx(x, y_pred)
        acyclicity_loss = self.compute_acyclicity_loss(input_layer_weights, compute_h_matrix_exp)
        sparsity_regularizer = tf.math.multiply(self.beta, self.compute_sparsity_loss(input_layer_weights))

        # Weight regularization losses
        regularization_loss = tf.math.add(reconstruction_loss, tf.math.add(acyclicity_loss, sparsity_regularizer),
                                          name="regularization_loss")

        weighted_regularization_loss = tf.math.multiply(self.lambda_weight, regularization_loss,
                                                        name="weighted_regularization_loss")

        loss = tf.math.add(prediction_loss, weighted_regularization_loss, name="overall_loss")

        # Update metrics
        self.metric_dict["loss_tracker"].update_state(loss)
        self.metric_dict["prediction_loss_tracker"].update_state(prediction_loss)
        self.metric_dict["reconstruction_loss_tracker"].update_state(reconstruction_loss)
        self.metric_dict["sparsity_loss_tracker"].update_state(sparsity_regularizer)
        self.metric_dict["acyclicity_loss_tracker"].update_state(acyclicity_loss)
        self.metric_dict["mse_x_tracker"].update_state(self.compute_mse_x(x, y_pred))
        self.metric_dict["mse_y_tracker"].update_state(self.compute_mse_y(y, y_pred))

        return loss

    def compute_l2_norm_matrix(self, input_layer_weights):
        """Compute matrix with L2-norms of input sub-layer weight matrices.
        Overrides base method.

        The entry [l2_norm_matrix]_(k,j) is the L2-norm of the k-th row of the weight matrix in input sub-layer j.
        Since our weight matrices are of dimension (d+1)x(d+1) (d is the number of x-variables, and we have d+1
        variables all together, x-variables and y), the resulting matrix will, have dimensions (d+1)x(d+1)

        Args:
            input_layer_weights (list of tensors): List with weight matrices of the input layers

        Returns:
            Tensor of shape (d+1)x(d+1), L2-norm matrix of input layer weights
        """
        l2_norm_matrix = list()
        for j, w in enumerate(input_layer_weights):
            l2_norm_matrix.append(tf.norm(w, axis=1, ord=2, name="l2_norm_input_layers"))
        return tf.stack(l2_norm_matrix, axis=1)

    def compute_sparsity_regularizer(self, input_layer_weights):
        """ Compute sparsity regularizer from the L1-norms of the input layer weights.
        Overrides base method.

        All input layers are masked and masked rows need to be accounted for when computing
        the sparsity loss.

        Args:
            input_layer_weights: (list of tensors): List with weight matrices of the input layers

        Returns:
            Tensor, sparsity regularizer value
        """
        sparsity_regularizer = 0.0
        for i, weight in enumerate(input_layer_weights):
            # Ignore the masked row
            w_1 = tf.slice(weight, [0, 0], [i, -1])
            w_2 = tf.slice(weight, [i + 1, 0], [-1, -1])

            sparsity_regularizer += tf.norm(w_1, ord=1, axis=[-2, -1], name="l1_norm_input_layers") \
                                    + tf.norm(w_2, ord=1, axis=[-2, -1], name="l1_norm_input_layers")
        return sparsity_regularizer

    @staticmethod
    def compute_mse_x(input_true, yx_pred):
        """Computes the MSE between input x-values and the predicted reconstructions.
        Overrides base method.

        `input_true` contains both y and x-values."""
        return tf.metrics.mse(input_true[:, 1:], yx_pred[:, 1:])

    def get_config(self):
        """Returns the config of `CASTLEOriginal`.
        Overrides base method.

       Config is a Python dictionary (serializable) containing the
       configuration of a `CASTLE` model. This allows
       the model to be re-instantiated later (without its trained weights)
       from this configuration.

       Note that `get_config()` does not guarantee to return a fresh copy of
       dict every time it is called. The callers should make a copy of the
       returned dict if they want to modify it.

       Returns:
           Python dictionary containing the configuration of `CASTLE`.
       """
        config = super(CASTLEOriginal, self).get_config()
        # These are the constructor arguments
        config.update(
            {
                "beta": self.beta,
                "lambda_weight": self.lambda_weight,
                "relu_alpha": self.relu_alpha,
            }
        )

        return config
