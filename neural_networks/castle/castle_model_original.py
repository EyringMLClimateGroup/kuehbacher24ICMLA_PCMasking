# Implementation for CASTLE neural network
# Paper: Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery. https://doi.org/10/grw6pt
# Original code at https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/castle and
# https://github.com/trentkyono/CASTLE
import tensorflow as tf

from neural_networks.castle.castle_model_base import CASTLEBase, build_graph, compute_h


@tf.keras.utils.register_keras_serializable()
class CASTLEOriginal(CASTLEBase):
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

    def __init__(self, num_inputs, hidden_layers, activation, rho, alpha, lambda_weight, relu_alpha=0.3, seed=None,
                 name="castle_original", **kwargs):
        num_input_layers = num_inputs + 1
        num_outputs = 1

        # Initialize the model as a functional (custom) model
        # Original CASTLE also passes Y as an input to the network. Use num_inputs + 1 as number of inputs
        input_sub_layers, shared_hidden_layers, output_sub_layers = build_graph(num_input_layers, num_inputs + 1,
                                                                                num_outputs, hidden_layers, relu_alpha,
                                                                                activation.lower(), seed)

        inputs = tf.keras.Input(shape=(num_inputs + 1,))
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
        outputs = tf.squeeze(tf.stack(yx_outputs, axis=1))

        super(CASTLEOriginal, self).__init__(num_inputs=num_inputs + 1, hidden_layers=hidden_layers,
                                             activation=activation, rho=rho, alpha=alpha, relu_alpha=relu_alpha,
                                             seed=seed, name=name, inputs=inputs,
                                             outputs=outputs, **kwargs)

        self.num_input_layers = num_input_layers
        self.lambda_weight = lambda_weight

        self.input_sub_layers = input_sub_layers
        self.shared_hidden_layers = shared_hidden_layers
        self.output_sub_layers = output_sub_layers

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
        input_layer_weights = [layer.trainable_variables[0] for layer in self.input_sub_layers]

        # In CASTLE, y_pred is (y_pred, x_pred)
        prediction_loss = self.compute_prediction_loss(y, y_pred)

        reconstruction_loss = self.compute_reconstruction_loss_yx(x, y_pred)
        acyclicity_loss = self.compute_acyclicity_loss(input_layer_weights, compute_h)
        sparsity_regularizer = self.compute_sparsity_loss(input_layer_weights)

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
                "num_inputs": self.num_inputs,
                "hidden_layers": self.hidden_layers,
                "activation": self.activation,
                "rho": self.rho,
                "alpha": self.alpha,
                "lambda_weight": self.lambda_weight,
                "relu_alpha": self.relu_alpha,
                "seed": self.seed,
            }
        )
        return config
