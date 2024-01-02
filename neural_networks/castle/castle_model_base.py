# Implementation for CASTLE neural network
# Paper: Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery. https://doi.org/10/grw6pt
# Original code at https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/castle and
# https://github.com/trentkyono/CASTLE
from abc import ABC

import keras.saving.serialization_lib
import tensorflow as tf
from tensorflow import keras


@tf.keras.utils.register_keras_serializable()
class CASTLEBase(tf.keras.Model, ABC):
    """Abstract base class for a neural network model with CASTLE (Causal Structure Learning) regularization
    adapted from Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery.
    https://doi.org/10/grw6pt.

    The output of the model is an array of shape `[batch_size, num_x_inputs + 1]`.
    The first element of the output (`output[:, 0]`) contains the prediction for the target variable `y`, while
    the other outputs are reconstructions of the regressors `x`.

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
        name (string) : Name of the model. Default: "castle_model".
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
                 name="castle_model", **kwargs):
        super(CASTLEBase, self).__init__(name=name, **kwargs)

        self.seed = seed

        self.activation = activation.lower()

        self.num_outputs = 1
        self.num_x_inputs = num_x_inputs

        self.hidden_layers = hidden_layers

        if kernel_initializer_input_layers is None:
            kernel_initializer_input_layers = {"initializers": "RandomNormal", "mean": 0.0, "std": 0.01}

        if kernel_initializer_hidden_layers is None:
            kernel_initializer_hidden_layers = {"initializers": "RandomNormal", "mean": 0.0, "std": 0.1}

        if kernel_initializer_output_layers is None:
            kernel_initializer_output_layers = {"initializers": "RandomNormal", "mean": 0.0, "std": 0.01}

        if bias_initializer_input_layers is None:
            bias_initializer_input_layers = "zeros"
        if bias_initializer_hidden_layers is None:
            bias_initializer_hidden_layers = "zeros"
        if bias_initializer_output_layers is None:
            bias_initializer_output_layers = "zeros"

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
        self.metric_dict["prediction_loss_tracker"] = tf.keras.metrics.Mean(name="prediction_loss")

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        """Compute the total loss and return it.

        CASTLEBase subclasses must override this method to provide custom loss
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
            "Unimplemented `neural_networks.castle.castle_model_base.CASTLEBase.compute_loss()`: "
            "You must subclass `CASTLEBase` with an overridden `compute_loss()` method.")

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
        """Returns the config of `CASTLEBase`.
        Overrides base method.

       Config is a Python dictionary (serializable) containing the
       configuration of a `CASTLEBase` model. This allows
       the model to be re-instantiated later (without its trained weights)
       from this configuration.

       Note that `get_config()` does not guarantee to return a fresh copy of
       dict every time it is called. The callers should make a copy of the
       returned dict if they want to modify it.

       Returns:
           Python dictionary containing the configuration of `CASTLE`.
       """
        config = super(CASTLEBase, self).get_config()
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


@tf.keras.utils.register_keras_serializable()
class CASTLEBaseWReconstruction(CASTLEBase):
    """Abstract base class for a neural network model with CASTLE (Causal Structure Learning) regularization
    adapted from Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery.
    https://doi.org/10/grw6pt.

    The output of the model is an array of shape `[batch_size, num_x_inputs + 1]`.
    The first element of the output (`output[:, 0]`) contains the prediction for the target variable `y`, while
    the other outputs are reconstructions of the regressors `x`.

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
        name (string) : Name of the model. Default: "castle_model".
        **kwargs: Keyword arguments.
    """

    def __init__(self,
                 num_x_inputs,
                 hidden_layers,
                 activation,
                 rho,
                 alpha,
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
                 name="castle_model", **kwargs):
        super(CASTLEBaseWReconstruction, self).__init__(num_x_inputs=num_x_inputs,
                                                        hidden_layers=hidden_layers,
                                                        activation=activation,
                                                        seed=seed,
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
                                                        name=name, **kwargs)

        # Set attributes
        self.rho = rho
        self.alpha = alpha

        self.num_input_layers = None

        self.input_sub_layers = None
        self.shared_hidden_layers = None
        self.output_sub_layers = None

        self.masks = list()

        # Metrics
        self.metric_dict = dict()
        self.metric_dict["loss_tracker"] = tf.keras.metrics.Mean(name="loss")
        self.metric_dict["prediction_loss_tracker"] = tf.keras.metrics.Mean(name="prediction_loss")
        self.metric_dict["reconstruction_loss_tracker"] = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.metric_dict["sparsity_loss_tracker"] = tf.keras.metrics.Mean(name="sparsity_loss")
        self.metric_dict["acyclicity_loss_tracker"] = tf.keras.metrics.Mean(name="acyclicity_loss")
        self.metric_dict["mse_x_tracker"] = tf.keras.metrics.Mean(name="mse_x")
        self.metric_dict["mse_y_tracker"] = tf.keras.metrics.Mean(name="mse_y")

    @staticmethod
    def compute_prediction_loss(y_true, yx_pred):  # y_true [batch,] yx_pred [batch, d+1, 1]
        """Computes CASTLE prediction loss."""
        return tf.reduce_mean(tf.keras.losses.mse(y_true, yx_pred[:, 0]),
                              name="prediction_loss_reduce_mean")

    @staticmethod
    def compute_reconstruction_loss_x(x_true, yx_pred):  # x_true [batch, d, 1] yx_pred [batch, d+1, 1]
        """Computes CASTLE reconstruction loss."""
        # Frobenius norm between all inputs and outputs averaged over the number of samples in the batch
        return tf.reduce_mean(
            tf.square(tf.norm(tf.subtract(tf.expand_dims(x_true, axis=-1), yx_pred[:, 1:]), ord='fro', axis=[-2, -1])),
            name="reconstruction_loss_reduce_mean")

    @staticmethod
    def compute_reconstruction_loss_yx(yx_true, yx_pred):
        """Computes CASTLE reconstruction loss."""
        # Frobenius norm between all inputs and outputs averaged over the number of samples in the batch
        return tf.reduce_mean(tf.norm(tf.subtract(tf.expand_dims(yx_true, axis=-1), yx_pred), ord='fro', axis=[-2, -1]),
                              name="reconstruction_loss_reduce_mean")

    def compute_acyclicity_loss(self, input_layer_weights, acyclicity_constraint_func, **kwargs):
        """Computes the Lagrangian optimization equation with the acyclicity constraint."""
        l2_norm_matrix = self.compute_l2_norm_matrix(input_layer_weights)

        h = compute_acyclicity_constraint(acyclicity_constraint_func, l2_norm_matrix, **kwargs)
        # tf.print(f"h function = {h}")

        # Acyclicity loss is computed using the Lagrangian scheme with penalty parameter rho and
        # Lagrangian multiplier alpha.
        h_squared = tf.math.square(h, name="h_squared")
        return tf.math.add(tf.math.multiply(tf.math.multiply(0.5, self.rho), h_squared, name="lagrangian_penalty"),
                           tf.math.multiply(self.alpha, h, name="lagrangian_optimizer"),
                           name="acyclicity_loss")

    def compute_l2_norm_matrix(self, input_layer_weights):
        """ Compute matrix with L2-norms of input sub-layer weight matrices.
        This method must be overridden in subclassed models."""
        raise NotImplementedError(
            "Unimplemented `neural_networks.castle.castle_model_base.CASTLEBase.compute_l2_norm_matrix()`: "
            "You must subclass `CASTLEBase` with an overridden `compute_l2_norm_matrix()` method.")

    @staticmethod
    def compute_sparsity_loss(input_layer_weights):
        """Computes sparsity loss as the sum of the matrix L1-norm of the input layer weight matrices."""
        sparsity_regularizer = 0.0

        # Entry-wise L1-norm (absolute sum of entries)
        for weight in input_layer_weights:
            entry_wise_norm = tf.norm(weight, ord=1, name='l1_norm_input_layers')
            # Scale by units of hidden layer (which is the number of columns in the matrix (transposed case))
            entry_wise_norm = tf.math.divide(entry_wise_norm, weight.shape[1], name="l1_norm_input_layers_scaled")
            sparsity_regularizer += entry_wise_norm

        # Scale with number of input layers
        sparsity_regularizer = sparsity_regularizer / len(input_layer_weights)
        return sparsity_regularizer

    @staticmethod
    def compute_mse_x(input_true, yx_pred):
        """Computes the MSE between input x-values and the predicted reconstructions.
        This method must be overridden in subclassed models."""
        raise NotImplementedError(
            "Unimplemented `neural_networks.castle.castle_model_base.CASTLEBase.compute_mse_x().`: "
            "You must subclass `CASTLEBase` with an overridden `compute_mse_x()` method.")

    @staticmethod
    def compute_mse_y(y_true, yx_pred):
        """Computes the MSE between the actual label y and its prediction."""
        return tf.metrics.mse(y_true, yx_pred[:, 0])

    def get_config(self):
        """Returns the config of `CASTLEBase`.
        Overrides base method.

       Config is a Python dictionary (serializable) containing the
       configuration of a `CASTLEBase` model. This allows
       the model to be re-instantiated later (without its trained weights)
       from this configuration.

       Note that `get_config()` does not guarantee to return a fresh copy of
       dict every time it is called. The callers should make a copy of the
       returned dict if they want to modify it.

       Returns:
           Python dictionary containing the configuration of `CASTLE`.
       """
        config = super(CASTLEBaseWReconstruction, self).get_config()
        # These are the constructor arguments
        config.update(
            {
                "rho": self.rho,
                "alpha": self.alpha
            }
        )
        return config


def compute_acyclicity_constraint(acyclicity_constraint_func, matrix, **kwargs):
    """
    Computes the acyclicity constraint for the given matrix with the
    function `acyclicity_constraint_func`.
    Keyword arguments for the `acyclicity_constraint_func` can be passed in `**kwargs`.

    Args:
        acyclicity_constraint_func (callable): Acyclicity constraint function.
        matrix (2d np.array or tf.Tensor): Matrix to compute the acyclicity constraint for.
        **kwargs: Keyword arguments for the acyclicity constraint function.

    Returns:
        Float value for acyclicity constraint.
    """
    if acyclicity_constraint_func is compute_h_matrix_exp:
        try:
            approximate = kwargs["approximate"]
            h = compute_h_matrix_exp(matrix, approximate=approximate)
        except KeyError:
            h = compute_h_matrix_exp(matrix)
    elif acyclicity_constraint_func is compute_h_log_det:
        try:
            s = kwargs["s"]
            h = compute_h_log_det(matrix, s=s)
        except KeyError:
            h = compute_h_log_det(matrix)
    else:
        raise ValueError(f"Unknown acyclicity constraint function: {acyclicity_constraint_func}.")
    return h


def compute_h_matrix_exp(matrix, approximate=True):
    """Compute the acyclicity constraint from NOTEARS for a (d x d)-matrix M:

        h(M) = tr(e^(M * M)) - d

    where `*` denotes the Hadamard (element-wise) product.

    See Zheng et al. (2018). DAGs with NO TEARS: Continuous Optimization for Structure Learning. https://doi.org/10/grxdgr
    and Zheng et al. (2019). Learning Sparse Nonparametric DAGs. https://doi.org/10/grxsr9
    for details.

    Args:
        matrix (tensor or numpy array): A matrix with shape with shape `[..., d, d]`.
        approximate (bool): If `True`, uses the same truncated power series as in original implementation
            from Kyono et al. 2020 to compute the matrix exponential. Otherwise, the Tensorflow function
            `tf.linalg.expm` is used for matrix exponential computation.

    Returns:
        float tensor: Value of the acyclicity constraint function.
    """
    d = matrix.shape[0]

    if approximate:
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


def compute_h_log_det(matrix, s=1.0):
    """Compute the acyclicity constraint function from DAGMA for a (d x d)-matrix W::

        h^s(W) = - logdet(sI - W * W) + d log(s)

    where `*` denotes the Hadamard (element-wise) product.

    See Bello et al. (2022). DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization.
        https://doi.org/10.48550/arXiv.2209.08037
    for details.

    Args:
        matrix (tensor or numpy array): A matrix with shape with shape `[..., d, d]`.
        s : float, optional
            Controls the domain of M-matrices. Defaults to 1.0.

    Returns:
        float tensor: Value of the acyclicity constraint function.
    """
    d = matrix.shape[0]

    id = tf.eye(d, dtype=tf.float32)

    m = s * id - matrix * matrix
    h = - tf.linalg.slogdet(m)[1] + d * tf.math.log(s)
    return h


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
