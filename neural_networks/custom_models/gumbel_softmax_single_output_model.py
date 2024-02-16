# Implementation for CASTLE neural network
# Paper: Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery. https://doi.org/10/grw6pt
# Original code at https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/castle and
# https://github.com/trentkyono/CASTLE
import numpy as np
import tensorflow as tf

from neural_networks.custom_models.layers.gumbel_softmax_layer import StraightThroughGumbelSoftmaxMaskingLayer
from neural_networks.custom_models.model_base import ModelBase, get_kernel_initializer
from utils.variable import Variable_Lev_Metadata


@tf.keras.utils.register_keras_serializable()
class GumbelSoftmaxSingleOutputModel(ModelBase):
    """A neural network model with for SPCAM parametrization prediction for a single output with masking
    using straight-through Gumbel-Softmax estimator ([1] and [2]).

    The masking vector is regularized using loss from semi-supervised image segmentation [3].

    Args:
        num_x_inputs (int): The number of regressors, i.e. the x-variables.
        hidden_layers (list of int): A list containing the hidden units for all hidden layers.
            ``len(hidden_layers)`` gives the number of hidden layers.
        activation (str, case insensitive): A string specifying the activation function,
            e.g. `relu`, `linear`, `sigmoid`, `tanh`. In addition to tf.keras specific strings for
            built-in activation functions, `LeakyReLU` can be used to specify leaky ReLU activation function.
            See also https://www.tensorflow.org/api_docs/python/tf/keras/layers/Activation.
        lambda_prediction (float): Weighting coefficient for prediction loss
        lambda_crf (float): Weighting coefficient for CRF regularization loss.
        lambda_vol_min (float): Weighting coefficient for minimum volume regularization loss.
        lambda_vol_avg (float): Weighting coefficient for average volume regularization loss.
        sigma_crf (float): Sigma value for CRF regularization loss.
        level_bins (list of int): List with bin separation values for binning vertical levels.
        output_var (Variable_Lev_Metadata or dict): SPCAM output variable for which the model is trained.
        ordered_input_vars (list of Variable_Lev_Metadata or dict): List of SPCAM input variables.
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


    [1] Jang, E., Gu, S., Poole, B., 2017. Categorical Reparameterization with Gumbel-Softmax.
        https://doi.org/10.48550/arXiv.1611.01144
    [2] Maddison, C.J., Mnih, A., Teh, Y.W., 2017. The Concrete Distribution: A Continuous Relaxation of
        Discrete Random Variables. https://doi.org/10.48550/arXiv.1611.00712
    [3] Veksler, O., 2020. Regularized Loss for Weakly Supervised Single Class Semantic Segmentation,
        Computer Vision – ECCV 2020. https://doi.org/10/gtgdj4
    """

    def __init__(self,
                 num_x_inputs,
                 hidden_layers,
                 activation,
                 lambda_prediction,
                 lambda_crf,
                 lambda_vol_min,
                 lambda_vol_avg,
                 sigma_crf,
                 level_bins,
                 output_var,
                 ordered_input_vars,
                 relu_alpha=0.3,
                 seed=None,
                 temperature=1.0,
                 kernel_initializer_input_layers=None,
                 kernel_initializer_hidden_layers=None,
                 kernel_initializer_output_layers=None,
                 bias_initializer_hidden_layers="zeros",
                 bias_initializer_output_layers="zeros",
                 kernel_regularizer_input_layers=None,
                 kernel_regularizer_hidden_layers=None,
                 kernel_regularizer_output_layers=None,
                 bias_regularizer_hidden_layers=None,
                 bias_regularizer_output_layers=None,
                 activity_regularizer_hidden_layers=None,
                 activity_regularizer_output_layers=None,
                 name="gumbel_softmax_single_output_model", **kwargs):

        num_outputs = 1

        # Create layers
        # Get activation function
        act_func = tf.keras.layers.LeakyReLU(alpha=relu_alpha) if activation.lower() == "leakyrelu" \
            else tf.keras.layers.Activation(activation.lower())

        input_layer = StraightThroughGumbelSoftmaxMaskingLayer(num_x_inputs,
                                                               temp=temperature,
                                                               params_initializer=kernel_initializer_input_layers,
                                                               params_regularizer=kernel_regularizer_input_layers,
                                                               seed=seed,
                                                               name="input_masking_layer")

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
        output_layer = tf.keras.layers.Dense(num_outputs, activation="linear", name=f"output_sub_layer",
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

        super(GumbelSoftmaxSingleOutputModel, self).__init__(num_x_inputs=num_x_inputs, hidden_layers=hidden_layers,
                                                             activation=activation, seed=seed,
                                                             kernel_initializer_input_layers=kernel_initializer_input_layers,
                                                             kernel_initializer_hidden_layers=kernel_initializer_hidden_layers,
                                                             kernel_initializer_output_layers=kernel_initializer_output_layers,
                                                             bias_initializer_hidden_layers=bias_initializer_hidden_layers,
                                                             bias_initializer_output_layers=bias_initializer_output_layers,
                                                             kernel_regularizer_input_layers=kernel_regularizer_input_layers,
                                                             kernel_regularizer_hidden_layers=kernel_regularizer_hidden_layers,
                                                             kernel_regularizer_output_layers=kernel_regularizer_output_layers,
                                                             bias_regularizer_hidden_layers=bias_regularizer_hidden_layers,
                                                             bias_regularizer_output_layers=bias_regularizer_output_layers,
                                                             activity_regularizer_hidden_layers=activity_regularizer_hidden_layers,
                                                             activity_regularizer_output_layers=activity_regularizer_output_layers,
                                                             name=name, inputs=inputs, outputs=outputs, **kwargs)

        self.relu_alpha = relu_alpha

        self.input_layer = input_layer
        self.shared_hidden_layers = shared_hidden_layers
        self.output_layer = output_layer

        self.level_bins = level_bins
        self.output_var = parse_variable_lev_metadata_to_dict(output_var)
        self.ordered_input_vars = [parse_variable_lev_metadata_to_dict(v) for v in ordered_input_vars]

        self.lambda_prediction = lambda_prediction
        self.lambda_crf = lambda_crf
        self.lambda_vol_min = lambda_vol_min
        self.lambda_vol_avg = lambda_vol_avg
        self.sigma_crf = sigma_crf

        self.vol_min = get_vol_min(self.output_var, self.level_bins)
        self.vol_avg = get_vol_avg(self.output_var, self.level_bins)

        # Additional metrics
        self.metric_dict["weighted_prediction_loss_tracker"] = tf.keras.metrics.Mean(name="weighted_prediction_loss")
        self.metric_dict["minimum_volume_loss_tracker"] = tf.keras.metrics.Mean(name="minimum_volume_loss")
        self.metric_dict["weighted_minimum_volume_loss_tracker"] = tf.keras.metrics.Mean(
            name="weighted_minimum_volume_loss")
        self.metric_dict["average_volume_loss_tracker"] = tf.keras.metrics.Mean(name="average_volume_loss")
        self.metric_dict["weighted_average_volume_loss_tracker"] = tf.keras.metrics.Mean(
            name="weighted_average_volume_loss")
        self.metric_dict["crf_loss_tracker"] = tf.keras.metrics.Mean(name="crf_loss")
        self.metric_dict["weighted_crf_loss_tracker"] = tf.keras.metrics.Mean(name="weighted_crf_loss_tracker")

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
        weighted_prediction_loss = self.lambda_prediction * prediction_loss

        crf_loss = self.compute_crf_loss(self.input_layer.masking_vector)
        weighted_crf_loss = self.lambda_crf * crf_loss

        min_vol_loss = self.compute_minimum_volume_loss(self.input_layer.masking_vector)
        weighted_min_vol_loss = self.lambda_vol_min * min_vol_loss

        avg_vol_loss = self.compute_average_volume_loss(self.input_layer.masking_vector)
        weighted_avg_vol_loss = self.lambda_vol_avg * avg_vol_loss

        loss = weighted_prediction_loss + weighted_crf_loss + weighted_min_vol_loss + weighted_avg_vol_loss

        # Update metrics
        self.metric_dict["loss_tracker"].update_state(loss)
        self.metric_dict["prediction_loss_tracker"].update_state(prediction_loss)
        self.metric_dict["weighted_prediction_loss_tracker"].update_state(weighted_prediction_loss)
        self.metric_dict["minimum_volume_loss_tracker"].update_state(min_vol_loss)
        self.metric_dict["weighted_minimum_volume_loss_tracker"].update_state(weighted_min_vol_loss)
        self.metric_dict["average_volume_loss_tracker"].update_state(avg_vol_loss)
        self.metric_dict["weighted_average_volume_loss_tracker"].update_state(weighted_avg_vol_loss)
        self.metric_dict["crf_loss_tracker"].update_state(crf_loss)
        self.metric_dict["weighted_crf_loss_tracker"].update_state(weighted_crf_loss)

        return loss

    @staticmethod
    def compute_prediction_loss(y_true, yx_pred):
        """Computes CASTLE prediction loss."""
        return tf.reduce_mean(tf.keras.losses.mse(y_true, yx_pred), name="prediction_loss_reduce_mean")

    def compute_minimum_volume_loss(self, masking_vector):
        min_vol_loss = tf.cast((tf.math.reduce_mean(masking_vector) < self.vol_min), tf.float32) * (
                tf.math.reduce_mean(masking_vector) - self.vol_min) ** 2
        return min_vol_loss

    def compute_average_volume_loss(self, masking_vector):
        avg_vol_loss = (tf.math.reduce_mean(masking_vector) - self.vol_avg) ** 2
        return avg_vol_loss

    def compute_crf_loss(self, masking_vector):
        crf_loss = tf.zeros_like(masking_vector)

        # Pre-calculate values outside the loop
        x_out = 1
        output_dim = self.output_var["dimensions"]

        if output_dim == 3:
            level_out = tf.cast(self.output_var["level"], tf.float32)

        for index, input_var in enumerate(self.ordered_input_vars):
            x_in = masking_vector[index]

            if input_var["dimensions"] == 3:
                level_in = tf.cast(input_var["level"], tf.float32)
            else:
                level_in = get_2d_level(input_var["name"])

            if output_dim == 2:
                level_out = get_2d_level(self.output_var["name"], level_in)

            crf_loss = tf.tensor_scatter_nd_update(
                crf_loss, [[index]],
                [tf.math.exp(-(level_out - level_in) ** 2 / (2 * self.sigma_crf)) * tf.abs(x_out - x_in)]
            )

        return tf.reduce_mean(crf_loss)

    def get_config(self):
        """Returns the config of `GumbelSoftmaxSingleOutputModel`.
        Overrides base method.

       Config is a Python dictionary (serializable) containing the
       configuration of a `GumbelSoftmaxSingleOutputModel` model. This allows
       the model to be re-instantiated later (without its trained weights)
       from this configuration.

       Note that `get_config()` does not guarantee to return a fresh copy of
       dict every time it is called. The callers should make a copy of the
       returned dict if they want to modify it.

       Returns:
           Python dictionary containing the configuration of `CASTLE`.
       """
        config = super(GumbelSoftmaxSingleOutputModel, self).get_config()
        # These are the constructor arguments

        config.update(
            {
                "lambda_prediction": self.lambda_prediction,
                "lambda_crf": self.lambda_crf,
                "lambda_vol_min": self.lambda_vol_min,
                "lambda_vol_avg": self.lambda_vol_avg,
                "sigma_crf": self.sigma_crf,
                "level_bins": self.level_bins,
                "output_var": self.output_var,
                "ordered_input_vars": self.ordered_input_vars,
                "relu_alpha": self.relu_alpha,
            }
        )

        return config


def get_bin(var, bins):
    if var["dimensions"] == 3:
        return np.digitize(var["level"], bins)
    else:
        return -1


def get_vol_avg(var, bins):
    b = get_bin(var, bins)

    if b == -1:
        # 2d case
        return 0.7

    if b == 0:
        # levels 0 - 100
        return 0.7  # physically, this should be 0, but we have SPCAM artifacts

    elif b == 1:
        # levels 101 - 440
        return 0.45

    elif b == 2:
        # levels 441 - 680
        return 0.75

    elif b == 3:
        # levels 681 - 1000
        return 0.45


def get_vol_min(var, bins):
    b = get_bin(var, bins)

    if b == -1:
        # 2d case
        return 0.3

    if b == 0:
        # levels 0 - 100
        return 0.05  # physically, this should be 0, but we have SPCAM artifacts

    elif b == 1:
        # levels 101 - 440
        return 0.1

    elif b == 2:
        # levels 441 - 680
        return 0.6

    elif b == 3:
        # levels 681 - 1000
        return 0.3


def get_2d_level(varname, in_level=None):
    # in
    if varname == "ps":
        return tf.cast(992, tf.float32)
    elif varname == "solin":
        return tf.cast(272, tf.float32)
    elif varname == "shflx":
        return tf.cast(912, tf.float32)
    elif varname == "lhflx":
        return tf.cast(912, tf.float32)
    # out
    elif varname == "fsnt":
        return tf.cast(3, tf.float32)
    elif varname == "fsns":
        return tf.cast(992, tf.float32)
    elif varname == "flnt":
        return tf.cast(3, tf.float32)
    elif varname == "flns":
        return tf.cast(992, tf.float32)
    elif varname == "prect":
        return tf.cast(in_level, tf.float32)
    else:
        raise ValueError(f"Level for 2d variable {varname} not known.")


def parse_variable_lev_metadata_to_dict(var):
    if isinstance(var, Variable_Lev_Metadata):
        var_dict = {
            "name": var.var.name,
            "level": var.level,
            "dimensions": var.var.dimensions
        }
    # else case assumes is already a correct dictionary (this is the case when reloading a model)
    else:
        var_dict = var

    return var_dict
