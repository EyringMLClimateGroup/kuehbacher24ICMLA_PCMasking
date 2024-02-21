# Implementation for CASTLE neural network
# Paper: Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery. https://doi.org/10/grw6pt
# Original code at https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/castle and
# https://github.com/trentkyono/CASTLE
import tensorflow as tf
from tensorflow import keras

from neural_networks.custom_models.model_base import CASTLEBaseWReconstruction, compute_h_matrix_exp, compute_h_log_det, \
    get_kernel_initializer
from neural_networks.custom_models.layers.masked_dense_layer import MaskedDenseLayer


@tf.keras.utils.register_keras_serializable()
class CASTLEAdapted(CASTLEBaseWReconstruction):
    """A neural network model with CASTLE (Causal Structure Learning) regularization adapted
    from Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery.
    https://doi.org/10/grw6pt.

    The output of the model is an array of shape `[batch_size, num_x_inputs + 1]`.
    The first element of the output (`output[:, 0]`) contains the prediction for the target variable `y`, while
    the other outputs are reconstructions of the regressors `x`.

    In contrast to the original paper, this adapted model only the receives the regressors `x` as input and not
    the label `y`. There is one input sub-layer for all elements in the input vector
    (i.e. there are `num_x_inputs` input sub-layers). The computation of the sparsity loss is also slightly
    adapted from the paper, as the L1-norm is scaled both by the number of units in the first hidden layer
    and the number of input layers.
    The L2-norms used to create the weight matrix for computing the acyclicity constraint
    are also scaled by the number of units in the first hidden layer.

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
        lambda_prediction (float): Weighting coefficient for prediction loss
        lambda_sparsity (float): Weighting coefficient for sparsity loss
        lambda_reconstruction (float): Weighting coefficient for reconstruction loss
        lambda_acyclicity (float): Weighting coefficient for acyclicity loss
        acyclicity_constraint (str, case insensitive): Specifies whether the acyclicity constraint from
            NOTEARS [1] or DAGMA [2] should be used.
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

    Raises:
         ValueError: If `acyclicity_constraint` is not in `['DAGMA', 'NOTEARS']`.


    [1] Zheng et al. 2018. DAGs with NO TEARS: Continuous Optimization for Structure Learning.
        https://doi.org/10/grxdgr
        Zheng et al. 2019. Learning Sparse Nonparametric DAGs. https://doi.org/10/grxsr9
    [2] Bello et al. 2022. DAGMA: Learning DAGs via M-matrices and a log-determinant acyclicity characterization.
        https://doi.org/10.48550/arXiv.2209.08037
    """

    def __init__(self,
                 num_x_inputs,
                 hidden_layers,
                 activation,
                 rho,
                 alpha,
                 lambda_prediction,
                 lambda_sparsity,
                 lambda_reconstruction,
                 lambda_acyclicity,
                 acyclicity_constraint,
                 dagma_s=1.0,
                 relu_alpha=0.3,
                 seed=None,
                 kernel_initializer_input_layers=None,
                 kernel_initializer_hidden_layers=None,
                 kernel_initializer_output_layers=None,
                 bias_initializer_input_layers="zeros",
                 bias_initializer_hidden_layers="zeros",
                 bias_initializer_output_layers="zeros",
                 kernel_regularizer_input_layers=None,
                 kernel_regularizer_hidden_layers=None,
                 kernel_regularizer_output_layers=None,
                 bias_regularizer_input_layers=None,
                 bias_regularizer_hidden_layers=None,
                 bias_regularizer_output_layers=None,
                 activity_regularizer_input_layers=None,
                 activity_regularizer_hidden_layers=None,
                 activity_regularizer_output_layers=None,
                 name="castle_adapted", **kwargs):
        num_input_layers = num_x_inputs + 1
        num_outputs = 1

        # Initialize the model as a functional (custom) model.

        # Create layers
        # Get activation function
        act_func = tf.keras.layers.LeakyReLU(alpha=relu_alpha) if activation == "leakyrelu" \
            else tf.keras.layers.Activation(activation)

        input_sub_layers = list()
        # In our adaptation, we don't pass Y as an input to the model
        # Therefore, the first input layer is not masked
        dense_layer = tf.keras.layers.Dense(hidden_layers[0], activation=act_func, name=f"input_sub_layer_0",
                                            kernel_initializer=get_kernel_initializer(kernel_initializer_input_layers,
                                                                                      seed),
                                            bias_initializer=bias_initializer_input_layers,
                                            kernel_regularizer=kernel_regularizer_input_layers,
                                            bias_regularizer=bias_regularizer_input_layers,
                                            activity_regularizer=activity_regularizer_input_layers)
        input_sub_layers.append(dense_layer)

        for i in range(num_input_layers - 1):
            mask = tf.transpose(
                tf.one_hot([i] * hidden_layers[0], depth=num_x_inputs, on_value=0.0, off_value=1.0, axis=-1))
            masked_dense_layer = MaskedDenseLayer(hidden_layers[0], mask, activation=act_func,
                                                  name=f"input_sub_layer_{i + 1}",
                                                  kernel_initializer=get_kernel_initializer(
                                                      kernel_initializer_input_layers, seed),
                                                  bias_initializer=bias_initializer_input_layers,
                                                  kernel_regularizer=kernel_regularizer_input_layers,
                                                  bias_regularizer=bias_regularizer_input_layers,
                                                  activity_regularizer=activity_regularizer_input_layers)
            input_sub_layers.append(masked_dense_layer)

        # Shared hidden layers: len(hidden_layers) number of hidden layers. All input sub-layers feed into the
        #   same (shared) hidden layers.
        shared_hidden_layers = list()
        for i, n_hidden_layer_nodes in enumerate(hidden_layers):
            shared_hidden_layers.append(
                tf.keras.layers.Dense(n_hidden_layer_nodes, activation=act_func, name=f"shared_hidden_layer_{i}",
                                      kernel_initializer=get_kernel_initializer(kernel_initializer_hidden_layers, seed),
                                      bias_initializer=bias_initializer_hidden_layers,
                                      kernel_regularizer=kernel_regularizer_hidden_layers,
                                      bias_regularizer=bias_regularizer_hidden_layers,
                                      activity_regularizer=activity_regularizer_hidden_layers))

        # Output sub-layers: One sub-layer for each input. Each output layer outputs one value, i.e.
        #   reconstructs one input.
        output_sub_layers = list()
        for i in range(num_input_layers):
            output_sub_layers.append(
                # No activation function, i.e. linear
                tf.keras.layers.Dense(num_outputs, activation="linear", name=f"output_sub_layer_{i}",
                                      kernel_initializer=get_kernel_initializer(kernel_initializer_output_layers, seed),
                                      bias_initializer=bias_initializer_output_layers,
                                      kernel_regularizer=kernel_regularizer_output_layers,
                                      bias_regularizer=bias_regularizer_output_layers,
                                      activity_regularizer=activity_regularizer_output_layers))

        inputs = tf.keras.Input(shape=(num_x_inputs,))
        # Create network graph
        inputs_hidden = [in_sub_layer(inputs) for in_sub_layer in input_sub_layers]

        # If there are no hidden layers, we go straight from input layers to output layers
        hidden_outputs = inputs_hidden
        for hidden_layer in shared_hidden_layers:
            hidden_outputs = list()
            # Pass all inputs through same hidden layers
            for x in inputs_hidden:
                hidden_outputs.append(hidden_layer(x))

            # Outputs become new inputs for next hidden layers
            inputs_hidden = hidden_outputs

        yx_outputs = [out_layer(x) for x, out_layer in zip(hidden_outputs, output_sub_layers)]

        # Stack outputs into one tensor
        # outputs = tf.squeeze(tf.stack(yx_outputs, axis=1))
        outputs = tf.stack(yx_outputs, axis=1, name="stack_outputs")
        # outputs = tf.reshape(outputs, shape=(tf.shape(outputs)[0], tf.shape(outputs)[1]), name="reshape_output_dims")

        super(CASTLEAdapted, self).__init__(num_x_inputs=num_x_inputs, hidden_layers=hidden_layers,
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

        self.lambda_prediction = lambda_prediction
        self.lambda_sparsity = lambda_sparsity
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_acyclicity = lambda_acyclicity

        self.relu_alpha = relu_alpha

        self.input_sub_layers = input_sub_layers
        self.shared_hidden_layers = shared_hidden_layers
        self.output_sub_layers = output_sub_layers

        self.acyclicity_constraint = acyclicity_constraint
        self.dagma_s = dagma_s
        if acyclicity_constraint.upper() == "NOTEARS":
            self.acyclicity_constraint_func = compute_h_matrix_exp
        elif acyclicity_constraint.upper() == "DAGMA":
            self.acyclicity_constraint_func = compute_h_log_det
        else:
            raise ValueError(f"Unknown value for acyclicity constraint function: {acyclicity_constraint}")

        # Additional metrics
        self.metric_dict["prediction_loss_tracker"] = tf.keras.metrics.Mean(name="prediction_loss")
        self.metric_dict["reconstruction_loss_tracker"] = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.metric_dict["sparsity_loss_tracker"] = tf.keras.metrics.Mean(name="sparsity_loss")
        self.metric_dict["acyclicity_loss_tracker"] = tf.keras.metrics.Mean(name="acyclicity_loss")
        self.metric_dict["weighted_prediction_loss_tracker"] = tf.keras.metrics.Mean(name="weighted_prediction_loss")
        self.metric_dict["weighted_reconstruction_loss_tracker"] = tf.keras.metrics.Mean(
            name="weighted_reconstruction_loss")
        self.metric_dict["weighted_sparsity_loss_tracker"] = tf.keras.metrics.Mean(name="weighted_sparsity_loss")
        self.metric_dict["weighted_acyclicity_loss_tracker"] = tf.keras.metrics.Mean(name="weighted_acyclicity_loss")
        self.metric_dict["mse_x_tracker"] = tf.keras.metrics.Mean(name="mse_x")
        self.metric_dict["mse_y_tracker"] = tf.keras.metrics.Mean(name="mse_y")

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        """
        Compute model loss. Overrides base method.

        Args:
            x: Input data.
            y: Target data.
            y_pred: Predictions returned by the model (output of `model(x)`). In the case of CASTLE,
              `y_pred` contains the prediction for `y` and the reconstruction of `x`.
            sample_weight: Sample weights for weighting the loss function.

        Returns:
            The total loss as a `tf.Tensor`, or `None` if no loss results (which
            is the case when called by `Model.test_step`).
        """
        # Todo: What is correct?
        input_layer_weights = [self.input_sub_layers[0].trainable_variables[0]]
        input_layer_weights.extend([layer.trainable_variables[0] * layer.mask for layer in self.input_sub_layers[1:]])
        # input_layer_weights = [layer.trainable_variables[0] for layer in self.input_sub_layers]

        # In CASTLE, y_pred is (y_pred, x_pred)
        prediction_loss = self.compute_prediction_loss(y, y_pred)
        weighted_prediction_loss = tf.math.multiply(self.lambda_prediction, prediction_loss)

        reconstruction_loss = self.compute_reconstruction_loss_x(x, y_pred)
        weighted_reconstruction_loss = tf.math.multiply(self.lambda_reconstruction, reconstruction_loss)
        acyclicity_loss = self.compute_acyclicity_loss(input_layer_weights, self.acyclicity_constraint_func,
                                                       s=self.dagma_s)
        weighted_acyclicity_loss = tf.math.multiply(self.lambda_acyclicity, acyclicity_loss)
        sparsity_regularizer = self.compute_sparsity_loss(input_layer_weights)
        weighted_sparsity_regularizer = tf.math.multiply(self.lambda_sparsity, sparsity_regularizer)

        # Regularization losses
        regularization_loss = tf.math.add(weighted_reconstruction_loss,
                                          tf.math.add(weighted_acyclicity_loss, weighted_sparsity_regularizer),
                                          name="regularization_loss")

        loss = tf.math.add(weighted_prediction_loss, regularization_loss, name="overall_loss")

        # Update metrics
        self.metric_dict["loss_tracker"].update_state(loss)
        self.metric_dict["prediction_loss_tracker"].update_state(prediction_loss)
        self.metric_dict["reconstruction_loss_tracker"].update_state(reconstruction_loss)
        self.metric_dict["sparsity_loss_tracker"].update_state(sparsity_regularizer)
        self.metric_dict["acyclicity_loss_tracker"].update_state(acyclicity_loss)
        self.metric_dict["weighted_prediction_loss_tracker"].update_state(weighted_prediction_loss)
        self.metric_dict["weighted_reconstruction_loss_tracker"].update_state(weighted_reconstruction_loss)
        self.metric_dict["weighted_sparsity_loss_tracker"].update_state(weighted_sparsity_regularizer)
        self.metric_dict["weighted_acyclicity_loss_tracker"].update_state(weighted_acyclicity_loss)
        self.metric_dict["mse_x_tracker"].update_state(self.compute_mse_x(x, y_pred))
        self.metric_dict["mse_y_tracker"].update_state(self.compute_mse_y(y, y_pred))

        return loss

    def compute_l2_norm_matrix(self, input_layer_weights):
        """ Compute matrix with L2-norms of input sub-layer weight matrices.
        Overrides base method.

        The entry [l2_norm_matrix]_(k,j) is the L2-norm of the k-th row of the weight matrix in input sub-layer j.
        Since our weight matrices are of dimension dxd (d is the number of x-variables), but we have d+1
        variables all together (x-variables and y) we set the first row 0 for y.

        Args:
            input_layer_weights (list of tensors): List with weight matrices of the input layers

        Returns:
            Tensor of shape (d+1)x(d+1), L2-norm matrix of input layer weights
        """
        # Norms should be in columns and first row should be zero
        l2_norm_matrix = list()
        for j, w in enumerate(input_layer_weights):
            l2_norm = tf.norm(w, axis=1, ord=2, name="l2_norm_input_layers")
            # Scale by units of hidden layer (which is the number of columns in the matrix (transposed case))
            l2_norm = tf.math.divide(l2_norm, w.shape[1], name="l2_norm_input_layers_scaled")
            l2_norm_matrix.append(tf.concat([tf.zeros((1,), dtype=tf.float32), l2_norm], axis=0))
        return tf.stack(l2_norm_matrix, axis=1)

    @staticmethod
    def compute_mse_x(input_true, yx_pred):
        """Computes the MSE between input x-values and the predicted reconstructions.
        Overrides base method.

        `input_true` contains only x-values."""
        return tf.metrics.mse(tf.expand_dims(input_true, axis=-1), yx_pred[:, 1:])

    def get_config(self):
        """Returns the config of `CASTLEAdapted`.
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
        config = super(CASTLEAdapted, self).get_config()
        # These are the constructor arguments
        config.update(
            {
                "lambda_prediction": self.lambda_prediction,
                "lambda_sparsity": self.lambda_sparsity,
                "lambda_reconstruction": self.lambda_reconstruction,
                "lambda_acyclicity": self.lambda_acyclicity,
                "acyclicity_constraint": self.acyclicity_constraint,
                "dagma_s": self.dagma_s,
                "relu_alpha": self.relu_alpha,
            }
        )
        return config
