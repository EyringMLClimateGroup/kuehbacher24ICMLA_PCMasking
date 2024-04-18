# Implementation for CASTLE neural network
# Paper: Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery. https://doi.org/10/grw6pt
# Original code at https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/castle and
# https://github.com/trentkyono/CASTLE
import tensorflow as tf
from tensorflow import keras

from neural_networks.custom_models.model_base import ModelBase, get_kernel_initializer


@tf.keras.utils.register_keras_serializable()
class PreMaskNet(ModelBase):
    def __init__(self,
                 num_x_inputs,
                 hidden_layers,
                 activation,
                 lambda_sparsity,
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
                 name="pre_mask_net", **kwargs):

        num_outputs = 1

        # Create layers
        # Get activation function
        act_func = tf.keras.layers.LeakyReLU(alpha=relu_alpha) if activation.lower() == "leakyrelu" \
            else tf.keras.layers.Activation(activation.lower())

        input_layer = tf.keras.layers.Dense(num_x_inputs, activation=act_func,
                                            name=f"input_layer",
                                            kernel_initializer=get_kernel_initializer(kernel_initializer_input_layers,
                                                                                      seed),
                                            bias_initializer=bias_initializer_input_layers,
                                            kernel_regularizer=kernel_regularizer_input_layers,
                                            bias_regularizer=bias_regularizer_input_layers,
                                            activity_regularizer=activity_regularizer_input_layers)

        # Hidden layers: len(hidden_layers) number of hidden layers
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
        output_layer = tf.keras.layers.Dense(num_outputs, activation="linear", name=f"output_layer",
                                             kernel_initializer=get_kernel_initializer(kernel_initializer_output_layers,
                                                                                       seed),
                                             bias_initializer=bias_initializer_output_layers,
                                             kernel_regularizer=kernel_regularizer_output_layers,
                                             bias_regularizer=bias_regularizer_output_layers,
                                             activity_regularizer=activity_regularizer_output_layers)
        # Create network graph
        inputs = tf.keras.Input(shape=(num_x_inputs,))
        inputs_hidden = input_layer(inputs)

        # If there are no hidden layers, we go straight from input layers to output layers
        hidden_outputs = inputs_hidden
        for hidden_layer in shared_hidden_layers:
            # Pass through hidden layer
            hidden_outputs = hidden_layer(inputs_hidden)

            # Outputs become new inputs for next hidden layers
            inputs_hidden = hidden_outputs

        outputs = output_layer(hidden_outputs)

        super(PreMaskNet, self).__init__(num_x_inputs=num_x_inputs, hidden_layers=hidden_layers,
                                         activation=activation, seed=seed,
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

        self.lambda_sparsity = lambda_sparsity
        self.relu_alpha = relu_alpha

        print(f"\n\nLambda sparsity = {self.lambda_sparsity}\n")

        self.input_layer = input_layer
        self.shared_hidden_layers = shared_hidden_layers
        self.output_layer = output_layer

        # Additional metrics
        self.metric_dict["prediction_loss_tracker"] = tf.keras.metrics.Mean(name="prediction_loss")
        self.metric_dict["sparsity_loss_tracker"] = tf.keras.metrics.Mean(name="sparsity_loss")
        self.metric_dict["weighted_sparsity_loss_tracker"] = keras.metrics.Mean(name="weighted_sparsity_loss")

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
        prediction_loss = self.compute_prediction_loss(y, y_pred)

        sparsity_regularizer = self.compute_sparsity_loss(self.input_layer.trainable_variables[0])
        weighted_sparsity_regularizer = tf.math.multiply(self.lambda_sparsity, sparsity_regularizer)

        loss = tf.math.add(prediction_loss, weighted_sparsity_regularizer, name="overall_loss")

        # Update metrics
        self.metric_dict["loss_tracker"].update_state(loss)
        self.metric_dict["prediction_loss_tracker"].update_state(prediction_loss)
        self.metric_dict["sparsity_loss_tracker"].update_state(sparsity_regularizer)
        self.metric_dict["weighted_sparsity_loss_tracker"].update_state(weighted_sparsity_regularizer)

        return loss

    @staticmethod
    def compute_prediction_loss(y_true, yx_pred):
        """Computes CASTLE prediction loss."""
        return tf.reduce_mean(tf.keras.losses.mse(y_true, yx_pred), name="prediction_loss_reduce_mean")

    @staticmethod
    def compute_sparsity_loss(input_layer_kernel):
        """Computes sparsity loss as the sum of the matrix L1-norm of the input layer weight matrices."""

        entry_wise_norm = tf.norm(input_layer_kernel, ord=1, name='l1_norm_input_layer')

        # Scale norm by matrix size (number of inputs * first hidden layer dimensions)
        sparsity_regularizer = tf.math.divide(entry_wise_norm,
                                              (input_layer_kernel.shape[0] * input_layer_kernel.shape[1]),
                                              name="l1_norm_input_layer_scaled")

        # matrix_norm = tf.norm(tf.transpose(input_layer_kernel), ord=2, axis=[-2, -1])
        # # Scale by number of rows (first hidden layer dimensions)
        # sparsity_regularizer = tf.divide(matrix_norm, input_layer_kernel.shape[1])

        return sparsity_regularizer

    def get_config(self):
        """Returns the config of `PreMaskNet`.
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
        config = super(PreMaskNet, self).get_config()
        # These are the constructor arguments
        config.update(
            {
                "lambda_sparsity": self.lambda_sparsity,
                "relu_alpha": self.relu_alpha,
            }
        )

        return config
