# Implementation for CASTLE neural network
# Paper: Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery. https://doi.org/10/grw6pt
# Original code at https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/castle and
# https://github.com/trentkyono/CASTLE
import numpy as np
import tensorflow as tf
from tensorflow import keras


# Todo:
#  - Implement partial training
#  - Implement CASTLE code version of loss
def build_castle(num_inputs, hidden_layers, activation, rho, alpha, lambda_, eager_execution=False,
                 hsic_prediction=False, strategy=None, seed=None):
    """
    Implement neural network with CASTLE (Causal Structure Learning) regularization
    from Kyono et al. 2020. CASTLE: Regularization via Auxiliary Causal Graph Discovery.
    https://doi.org/10/grw6pt.

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
        lambda_ (float): Weighting coefficient for the regularization term in the training loss.
        eager_execution (bool): If `True`, the code will be executed eagerly and the model's logic will
            not be wrapped inside a tf.function. Can be used for debugging purposes. Defaults to `False`.
        hsic_prediction (bool):
        seed (int): Random seed. Used to make the behavior of the initializer deterministic.
            Note that a seeded initializer will produce the same random values across multiple calls.

    Returns:
        tf.keras.Model: A tf.keras model designed according to CASTLE architecture.

    Raises:
        ValueError: If `rho` is not greater than 0.
    """
    # Enable eager execution for debugging
    tf.config.run_functions_eagerly(eager_execution)
    # Force eager execution of tf.data functions as well
    if eager_execution:
        tf.data.experimental.enable_debug_mode()

    if rho <= 0:
        raise ValueError("Penalty parameter `rho` for Lagrangian optimization scheme for acyclicity constraint "
                         "must be greater than 0.")

    # Can only predict for single y
    num_outputs = 1

    def _build_castle():
        input_sub_layers, inputs, model_, outputs = _create_model(activation, hidden_layers, num_inputs, num_outputs,
                                                                  seed)
        # Add MSE metric to model
        model_.add_metric(tf.metrics.mse(tf.expand_dims(inputs, axis=-1), tf.transpose(outputs[1:], [1, 0, 2])),
                          name="mse_x")

        # Compile model
        loss_func = castle_loss([l.get_weights()[0] for l in input_sub_layers], rho, alpha, lambda_)
        return _compile_castle(model_, loss_func, eager_execution)

    if strategy is not None:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = _build_castle()

    else:
        model = _build_castle()
    return model


def _create_model(activation, hidden_layers, nn_inputs, num_outputs, seed):
    # Get activation function
    activation = activation.lower()
    act = tf.keras.layers.LeakyReLU(alpha=0.3) if activation == "leakyrelu" else tf.keras.layers.Activation(activation)

    # 1. Create layers
    # Neural net inputs
    inputs = keras.Input(shape=(nn_inputs,), name="input_tensor")

    # The number of input sub-layers is nn_inputs + 1, because we need one sub-layer with all
    # x-variables for the prediction of y, and nn_input layers for the prediction for each of the inputs
    num_input_layers = nn_inputs + 1

    # Input sub-layers: One sub-layer for each input and each sub-layers receives all the inputs
    # We're using RandomNormal initializers for kernel and bias because the original CASTLE
    # implementation used random normal initialization.
    # Todo: Check the standard deviation for the input layers. In CASTLE it's initialized as
    #  tf.Variable(tf.random_normal([self.n_hidden], seed=seed) * 0.01)
    #  and default values are mean=0.0, stddev=1.0
    input_sub_layers = list()
    for i in range(num_input_layers):
        input_sub_layers.append(
            keras.layers.Dense(hidden_layers[0], activation=act, name=f"input_sub_layer_{i}",
                               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=seed),
                               bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=seed)))

    # Shared hidden layers: len(hidden_layers) number of hidden layers. All input sub-layers feed into the
    #   same (shared) hidden layers.
    shared_hidden_layers = list()
    for i, n_hidden_layer_nodes in enumerate(hidden_layers):
        shared_hidden_layers.append(
            keras.layers.Dense(n_hidden_layer_nodes, activation=act, name=f"shared_hidden_layer_{i}",
                               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=seed),
                               bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=seed)))

    # Output sub-layers: One sub-layer for each input. Each output layer outputs one value, i.e.
    #   reconstructs one input.
    output_sub_layers = list()
    for i in range(num_input_layers):
        output_sub_layers.append(
            # No activation function, i.e. linear
            keras.layers.Dense(num_outputs, activation="linear", name=f"output_sub_layer_{i}",
                               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=seed),
                               bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=seed)))

    # 2. Create network graph
    inputs_hidden = [in_sub_layer(inputs) for in_sub_layer in input_sub_layers]
    hidden_outputs = list()
    for hidden_layer in shared_hidden_layers:
        # Pass all inputs through same hidden layers
        for x in inputs_hidden:
            hidden_outputs.append(hidden_layer(x))

        # Outputs become new inputs for next hidden layers
        inputs_hidden = hidden_outputs[-num_input_layers:]

    outputs = [out_layer(x) for x, out_layer in zip(hidden_outputs[-num_input_layers:], output_sub_layers)]
    # Stack the outputs into one tensor
    outputs = tf.stack(outputs)

    # 3. Mask matrix rows in input layers so that a variable cannot cause itself
    #    Since the first input sub-layer aims to use all the x-variables to predict y, it does not need a mask.
    #    For the other sub-layers, we need to mask the x-variable that is to be predicted.
    # This can only be done after the graph is created because the layer weights are initialized only then
    for i in range(num_input_layers - 1):
        weights, bias = input_sub_layers[i + 1].get_weights()
        # Create one hot matrix that masks the row of the current input
        mask = tf.transpose(tf.one_hot([i] * hidden_layers[0], depth=nn_inputs, on_value=0.0, off_value=1.0, axis=-1))
        input_sub_layers[i + 1].set_weights([weights * mask, bias])

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='castleNN')
    return input_sub_layers, inputs, model, outputs


def castle_loss(input_layer_weights, rho, alpha, reg_lambda):
    # function that accepts the ground truth and predictions
    def _castle_loss(y_true, y_pred):
        # y_true shape [batch_size, num_x + 1]
        # y_pred shape [num_x + 1, batch_size, 1]

        # Compute loss and add to model.
        # CASTLE loss consists of four components:
        # overall_loss = prediction_loss + lambda * regularization_loss
        # where regularization_loss = reconstruction_loss + acyclicity_loss + sparsity_regularization

        # Prediction loss
        prediction_loss = tf.reduce_mean(keras.losses.mse(y_pred[0], y_true[:, 0]), name="prediction_loss_reduce_mean")
        # tf.print(f"\n\nprediction loss = {prediction_loss}")

        # Reconstruction loss
        # Frobenius norm between all inputs and outputs averaged over the number of samples in the batch
        reconstruction_loss = tf.reduce_mean(
            tf.norm(y_true[:, 1:] - tf.transpose(y_pred[1:], [1, 0]), ord='fro', axis=[-2, -1]),
            name="reconstruction_loss_reduce_mean")
        # tf.print(f"reconstruction loss = {reconstruction_loss}")

        # Acyclicity loss
        # Compute matrix with l2-norms of input sub-layer weight matrices:
        # The entry [l2_norm_matrix]_(k,j) is the l2-norm of the k-th row of the weight matrix in input sub-layer j.
        # Since our weight matrices are of dimension dxd (d is the number of x-variables), but we have d+1
        # variables all together (x-variables and y) we set the first column 0 for y.
        l2_norm_matrix = list()
        for j, w in enumerate(input_layer_weights):
            l2_norm_matrix.append(tf.concat([tf.zeros((1,), dtype=tf.float32),
                                             tf.norm(w, axis=1, ord=2, name="l2_norm_input_layers")], axis=0))
        l2_norm_matrix = tf.stack(l2_norm_matrix, axis=1)
        # tf.print(f"l2 norm matrix = {l2_norm_matrix}")

        h = compute_h(l2_norm_matrix)
        # tf.print(f"h function = {h}")

        # Acyclicity loss is computed using the Lagrangian scheme with penalty parameter rho and
        # Lagrangian multiplier alpha.
        h_squared = tf.math.square(h, name="h_squared")
        acyclicity_loss = tf.math.add(tf.math.multiply(0.5 * rho, h_squared, name="lagrangian_penalty"),
                                      tf.math.multiply(alpha, h, name="lagrangian_optimizer"),
                                      name="acyclicity_loss")
        # tf.print(f"acyclicity loss = {acyclicity_loss}")

        # Sparsity regularizer
        # Compute the l1-norm for the weight matrices in th einput_sublayer
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
        # tf.print(f"sparsity regularizer = {sparsity_regularizer}")

        # Add everything up to form overall loss
        regularization_loss = tf.math.add(reconstruction_loss, tf.math.add(acyclicity_loss, sparsity_regularizer),
                                          name="regularization_loss")
        # tf.print(f"regularization loss = {regularization_loss}\n\n")
        overall_loss = tf.math.add(prediction_loss,
                                   tf.math.multiply(reg_lambda, regularization_loss, name="weighted_regularization"),
                                   name="overall_loss")
        return overall_loss

    # return the inner function tuned by the hyperparameter
    return _castle_loss


def _compile_castle(model, loss_func, eager_execution):
    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
        jit_compile=True
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_func,
        run_eagerly=eager_execution,
    )

    return model


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
