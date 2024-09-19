import keras.saving.serialization_lib
import tensorflow as tf

from pcmasking.neural_networks.custom_models.model_base import ModelBase, get_kernel_initializer


@tf.keras.utils.register_keras_serializable()
class MaskNet(ModelBase):
    """Tensorflow model for training a PCMasking framework network in masking mode.

    Args:
        num_x_inputs (int): The number inputs (the observed variables x).
        hidden_layers (list of int): A list containing the hidden units for all hidden layers.
            ``len(hidden_layers)`` gives the number of hidden layers.
        activation (str, case insensitive): A string specifying the activation function,
            e.g. `relu`, `linear`, `sigmoid`, `tanh`. In addition to tf.keras specific strings for
            built-in activation functions, `LeakyReLU` can be used to specify leaky ReLU activation function.
            See also https://www.tensorflow.org/api_docs/python/tf/keras/layers/Activation.
        masking_vector (tf.Tensor or np.array): Masking vector.
        threshold (float): Threshold for masking vector.
        relu_alpha (float): Negative slope coefficient for leaky ReLU activation function. Default: 0.3.
        seed (int): Random seed.
        kernel_initializer_hidden_layers (tf.keras.initializers.Initializer): Initializer for the
            weight matrix of the dense hidden layers.
        kernel_initializer_output_layers (tf.keras.initializers.Initializer): Initializer for the
            weight matrix of the dense output layer.
        bias_initializer_hidden_layers (tf.keras.initializers.Initializer): Initializer for the bias vector
            of the dense hidden layers.
        bias_initializer_output_layers (tf.keras.initializers.Initializer): Initializer for the bias vector
            of the dense output layer.
        kernel_regularizer_hidden_layers (tf.keras.regularizers.Regularizer): Regularizer function applied
            to the weight matrix of the dense hidden layers.
        kernel_regularizer_output_layers (tf.keras.regularizers.Regularizer): Regularizer function applied
            to the weight matrix of the dense output layer.
        bias_regularizer_hidden_layers (tf.keras.regularizers.Regularizer): Regularizer function applied
            to the bias vector of the dense hidden layers.
        bias_regularizer_output_layers (tf.keras.regularizers.Regularizer): Regularizer function applied
            to the bias vector of the dense output layer.
        activity_regularizer_hidden_layers (tf.keras.regularizers.Regularizer): Regularizer function applied
             to the output of the dense hidden layers (its "activation").
        activity_regularizer_output_layers (tf.keras.regularizers.Regularizer): Regularizer function applied
             to the output of the dense output layer (its "activation").
        name (string): Name of the model. Default: "mask_net".
        **kwargs: Keyword arguments
    """

    def __init__(self,
                 num_x_inputs,
                 hidden_layers,
                 activation,
                 masking_vector,
                 threshold,
                 relu_alpha=0.3,
                 seed=None,
                 kernel_initializer_hidden_layers=None,
                 kernel_initializer_output_layers=None,
                 bias_initializer_hidden_layers="zeros",
                 bias_initializer_output_layers="zeros",
                 kernel_regularizer_hidden_layers=None,
                 kernel_regularizer_output_layers=None,
                 bias_regularizer_hidden_layers=None,
                 bias_regularizer_output_layers=None,
                 activity_regularizer_hidden_layers=None,
                 activity_regularizer_output_layers=None,
                 name="mask_net", **kwargs):
        num_outputs = 1

        # Masking vector
        masking_vector = tf.Variable(masking_vector, trainable=False, name="masking_vector")

        # Threshold masking vector
        masking_vector.assign(tf.where(masking_vector > threshold,
                                       x=tf.ones_like(masking_vector),
                                       y=tf.zeros_like(masking_vector),
                                       name="threshold_masking_vector"))
        # Create layers
        # Get activation function
        act_func = tf.keras.layers.LeakyReLU(alpha=relu_alpha) if activation.lower() == "leakyrelu" \
            else tf.keras.layers.Activation(activation.lower())

        # Hidden layers: len(hidden_layers) number of hidden layers
        shared_hidden_layers = list()
        for i, n_hidden_layer_nodes in enumerate(hidden_layers):
            shared_hidden_layers.append(
                tf.keras.layers.Dense(n_hidden_layer_nodes, activation=act_func, name=f"hidden_layer_{i}",
                                      kernel_initializer=get_kernel_initializer(kernel_initializer_hidden_layers, seed),
                                      bias_initializer=bias_initializer_hidden_layers,
                                      kernel_regularizer=kernel_regularizer_hidden_layers,
                                      bias_regularizer=bias_regularizer_hidden_layers,
                                      activity_regularizer=activity_regularizer_hidden_layers))

        # Output sub-layers: One sub-layer for each input. Each output layer outputs one value, i.e.
        #   reconstructs one input.
        output_layer = tf.keras.layers.Dense(num_outputs, activation="linear", name=f"output_sub_layer",
                                             kernel_initializer=get_kernel_initializer(kernel_initializer_output_layers,
                                                                                       seed),
                                             bias_initializer=bias_initializer_output_layers,
                                             kernel_regularizer=kernel_regularizer_output_layers,
                                             bias_regularizer=bias_regularizer_output_layers,
                                             activity_regularizer=activity_regularizer_output_layers)
        # Create network graph
        inputs = tf.keras.Input(shape=(num_x_inputs,), name="inputs")
        inputs_hidden = tf.math.multiply(inputs, masking_vector, name="mask_inputs")

        # If there are no hidden layers, we go straight from input layers to output layers
        hidden_outputs = inputs_hidden
        for hidden_layer in shared_hidden_layers:
            # Pass through hidden layer
            hidden_outputs = hidden_layer(inputs_hidden)

            # Outputs become new inputs for next hidden layers
            inputs_hidden = hidden_outputs

        outputs = output_layer(hidden_outputs)

        # We have to initialize this way in order for Tensorflow to treat the model as subclassed
        # function model (which it only realizes when super.__init__ is called with input and output tensor)
        super(MaskNet, self).__init__(num_x_inputs=num_x_inputs, hidden_layers=hidden_layers,
                                      activation=activation, seed=seed,
                                      kernel_initializer_hidden_layers=kernel_initializer_hidden_layers,
                                      kernel_initializer_output_layers=kernel_initializer_output_layers,
                                      bias_initializer_hidden_layers=bias_initializer_hidden_layers,
                                      bias_initializer_output_layers=bias_initializer_output_layers,
                                      kernel_regularizer_hidden_layers=kernel_regularizer_hidden_layers,
                                      kernel_regularizer_output_layers=kernel_regularizer_output_layers,
                                      bias_regularizer_hidden_layers=bias_regularizer_hidden_layers,
                                      bias_regularizer_output_layers=bias_regularizer_output_layers,
                                      activity_regularizer_hidden_layers=activity_regularizer_hidden_layers,
                                      activity_regularizer_output_layers=activity_regularizer_output_layers,
                                      name=name, inputs=inputs, outputs=outputs, **kwargs)

        self.relu_alpha = relu_alpha
        self.threshold = threshold

        self.masking_vector = masking_vector
        self.shared_hidden_layers = shared_hidden_layers
        self.output_layer = output_layer

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        """
        Compute model loss. Overrides base method.

        Args:
            x: Input data.
            y: Target data.
            y_pred: Predictions returned by the model (output of `model(x)`).
            sample_weight: Sample weights for weighting the loss function.

        Returns:
            The total loss as a `tf.Tensor`, or `None` if no loss results (which
            is the case when called by `Model.test_step`).
        """
        prediction_loss = self.compute_prediction_loss(y, y_pred)

        # Update metrics
        self.metric_dict["loss_tracker"].update_state(prediction_loss)

        return prediction_loss

    @staticmethod
    def compute_prediction_loss(y_true, yx_pred):
        """Computes prediction loss."""
        return tf.reduce_mean(tf.keras.losses.mse(y_true, yx_pred), name="prediction_loss_reduce_mean")

    def get_config(self):
        """Returns the config of `MaskNet`.
        Overrides base method.

        Config is a Python dictionary (serializable) containing theconfiguration of a `MaskNet` model.
        This allows the model to be re-instantiated later (without its trained weights)
        from this configuration.

        Returns:
            Python dictionary containing the configuration of `MaskNet`.
       """
        config = super(MaskNet, self).get_config()
        # These are the constructor arguments
        config.update(
            {
                "relu_alpha": self.relu_alpha,
                "masking_vector": keras.saving.serialization_lib.serialize_keras_object(self.masking_vector.numpy()),
                "threshold": self.threshold,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        """Creates a model from its config.

       This method is the reverse of `get_config`, capable of instantiating the same model
       from the config dictionary.

       Args:
           config: A Python dictionary, typically the output of get_config.

       Returns:
           A model instance.
        """
        masking_vector = config.pop("masking_vector")
        masking_vector = keras.saving.serialization_lib.deserialize_keras_object(masking_vector)
        return cls(**config, masking_vector=masking_vector)
