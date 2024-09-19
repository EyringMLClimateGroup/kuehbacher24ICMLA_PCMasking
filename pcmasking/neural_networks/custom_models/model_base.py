from abc import ABC

import keras.saving.serialization_lib
import tensorflow as tf
from tensorflow import keras


@tf.keras.utils.register_keras_serializable()
class ModelBase(tf.keras.Model, ABC):
    """Abstract base class for custom neural network model.

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
        name (string) : Name of the model. Default: "model_base".
        **kwargs: Keyword arguments.
    """

    def __init__(self,
                 num_x_inputs,
                 hidden_layers,
                 activation,
                 seed=None,
                 kernel_initializer_input_layers=None,
                 kernel_initializer_hidden_layers=None,
                 kernel_initializer_output_layers=None,
                 bias_initializer_input_layers='zeros',
                 bias_initializer_hidden_layers='zeros',
                 bias_initializer_output_layers='zeros',
                 kernel_regularizer_input_layers=None,
                 kernel_regularizer_hidden_layers=None,
                 kernel_regularizer_output_layers=None,
                 bias_regularizer_input_layers=None,
                 bias_regularizer_hidden_layers=None,
                 bias_regularizer_output_layers=None,
                 activity_regularizer_input_layers=None,
                 activity_regularizer_hidden_layers=None,
                 activity_regularizer_output_layers=None,
                 name="model_base", **kwargs):
        super(ModelBase, self).__init__(name=name, **kwargs)

        self.seed = seed

        self.activation = activation.lower()

        self.num_outputs = 1
        self.num_x_inputs = num_x_inputs

        self.hidden_layers = hidden_layers

        if kernel_initializer_input_layers is None:
            kernel_initializer_input_layers = {"initializers": "GlorotUniform"}

        if kernel_initializer_hidden_layers is None:
            kernel_initializer_hidden_layers = {"initializers": "GlorotUniform"}

        if kernel_initializer_output_layers is None:
            kernel_initializer_output_layers = {"initializers": "GlorotUniform"}

        self.kernel_initializer_input_layers = kernel_initializer_input_layers
        self.kernel_initializer_hidden_layers = kernel_initializer_hidden_layers
        self.kernel_initializer_output_layers = kernel_initializer_output_layers

        self.bias_initializer_input_layers = bias_initializer_input_layers
        self.bias_initializer_hidden_layers = bias_initializer_hidden_layers
        self.bias_initializer_output_layers = bias_initializer_output_layers

        self.kernel_regularizer_input_layers = kernel_regularizer_input_layers
        self.kernel_regularizer_hidden_layers = kernel_regularizer_hidden_layers
        self.kernel_regularizer_output_layers = kernel_regularizer_output_layers

        self.bias_regularizer_input_layers = bias_regularizer_input_layers
        self.bias_regularizer_hidden_layers = bias_regularizer_hidden_layers
        self.bias_regularizer_output_layers = bias_regularizer_output_layers

        self.activity_regularizer_input_layers = activity_regularizer_input_layers
        self.activity_regularizer_hidden_layers = activity_regularizer_hidden_layers
        self.activity_regularizer_output_layers = activity_regularizer_output_layers

        # Metrics
        self.metric_dict = dict()
        self.metric_dict["loss_tracker"] = tf.keras.metrics.Mean(name="loss")

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        """Compute the total loss and return it.

        ModelBase subclasses must override this method to provide custom loss
        computation logic.

        Args:
          x: Input data.
          y: Target data.
          y_pred: Predictions returned by the model (output of `model(x)`)
          sample_weight: Sample weights for weighting the loss function.

        Returns:
          The total loss as a `tf.Tensor`, or `None` if no loss results (which
          is the case when called by `Model.test_step`).
        """
        raise NotImplementedError(
            "Unimplemented `pcmasking.neural_networks.custom_models.model_base.ModelBase.compute_loss()`: "
            "You must subclass `ModelBase` with an overridden `compute_loss()` method.")

    def reset_metrics(self):
        """Resets the state of all the metrics in the model."""
        for metric in self.metric_dict.values():
            metric.reset_state()

    @property
    def metrics(self):
        """Returns a list of model metrics."""
        # We list our `Metric` objects here so that `reset_state()` can be
        # called automatically at the start of each epoch or at the start of `evaluate()`..
        return list(self.metric_dict.values())

    def get_config(self):
        """Returns the config of `ModelBase`.
        Overrides base method.

       Config is a Python dictionary (serializable) containing the
       configuration of a `ModelBase` model. This allows
       the model to be re-instantiated later (without its trained weights)
       from this configuration.

       Note that `get_config()` does not guarantee to return a fresh copy of
       dict every time it is called. The callers should make a copy of the
       returned dict if they want to modify it.

       Returns:
           Python dictionary containing the configuration of `ModelBase`.
       """
        config = super(ModelBase, self).get_config()
        # These are the constructor arguments
        config.update(
            {
                "num_x_inputs": self.num_x_inputs,
                "hidden_layers": self.hidden_layers,
                "activation": self.activation,

                "kernel_initializer_input_layers": self.kernel_initializer_input_layers,
                "kernel_initializer_hidden_layers": self.kernel_initializer_hidden_layers,
                "kernel_initializer_output_layers": self.kernel_initializer_output_layers,

                "bias_initializer_input_layers": self.bias_initializer_input_layers,
                "bias_initializer_hidden_layers": self.bias_initializer_hidden_layers,
                "bias_initializer_output_layers": self.bias_initializer_output_layers,

                "kernel_regularizer_input_layers": self.kernel_regularizer_input_layers,
                "kernel_regularizer_hidden_layers": self.kernel_regularizer_hidden_layers,
                "kernel_regularizer_output_layers": self.kernel_regularizer_output_layers,

                "bias_regularizer_input_layers": self.bias_regularizer_input_layers,
                "bias_regularizer_hidden_layers": self.bias_regularizer_hidden_layers,
                "bias_regularizer_output_layers": self.bias_regularizer_output_layers,

                "activity_regularizer_input_layers": self.activity_regularizer_input_layers,
                "activity_regularizer_hidden_layers": self.activity_regularizer_hidden_layers,
                "activity_regularizer_output_layers": self.activity_regularizer_output_layers,

                "seed": self.seed,
            }
        )
        return config


def get_kernel_initializer(kernel_initializer, seed):
    """
    Parses the kernel initializer from given string to tf.keras.initializers.Initializer instance.

    Args:
        kernel_initializer (str): String specifying kernel initializer type.
        seed (int): Random seed for kernel initializer. Used to make the behavior of the initializer
            deterministic. Note that a seeded initializer will not produce the same random values across
            multiple calls, but multiple initializers will produce the same sequence when
            constructed with the same seed value.

    Returns:
        tf.keras.initializers.Initializer: kernel initializer instance

    Raises:
        ValueError: If `kernel_initializer` is not in `['Constant', 'GlorotNormal', 'GlorotUniform',
            'HeNormal', 'HeUniform', 'Identity', 'LecunNormal', 'LecunUniform', 'Ones', 'Orthogonal',
            'RandomNormal', 'RandomUniform','TruncatedNormal', 'VarianceScaling', 'Zeros']`.
    """
    # Support legacy saved models where the initializer was saved as keras initializer
    if isinstance(kernel_initializer, keras.initializers.Initializer):
        return kernel_initializer

    if kernel_initializer is None:
        kernel_initializer = keras.initializers.RandomNormal(mean=0.0,
                                                             stddev=0.01, seed=seed)
    elif isinstance(kernel_initializer, str):
        kernel_initializer = keras.initializers.get(kernel_initializer)
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
