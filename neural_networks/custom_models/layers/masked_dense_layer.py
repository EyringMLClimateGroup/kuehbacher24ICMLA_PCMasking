import tensorflow as tf
from tensorflow import keras


@tf.keras.utils.register_keras_serializable()
class MaskedDenseLayer(keras.layers.Layer):
    def __init__(self, units, mask, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(
                "Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )
        self.mask = tf.Variable(mask, trainable=False)
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "A Dense layer can only be built with a floating-point "
                f"dtype. Received: dtype={dtype}"
            )

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )
        if self.mask.shape != (last_dim, self.units):
            raise ValueError(
                "The dimensions of mask and kernel must match. "
                f"Kernel shape: {(last_dim, self.units)}. "
                f"Mask shape: {self.mask.shape}"
            )
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[
                    self.units,
                ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        self.built = True

        # Mask kernel
        weight_bias = self.get_weights()
        weight_bias[0] = weight_bias[0] * self.mask
        self.set_weights(weight_bias)

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        is_ragged = isinstance(inputs, tf.RaggedTensor)
        if is_ragged:
            # In case we encounter a RaggedTensor with a fixed last dimension
            # (last dimension not ragged), we can flatten the input and restore
            # the ragged dimensions at the end.
            if tf.compat.dimension_value(inputs.shape[-1]) is None:
                raise ValueError(
                    "Dense layer only supports RaggedTensors when the "
                    "innermost dimension is non-ragged. Received: "
                    f"inputs.shape={inputs.shape}."
                )
            original_inputs = inputs
            if inputs.flat_values.shape.rank > 1:
                inputs = inputs.flat_values
            else:
                # Innermost partition is encoded using uniform_row_length.
                # (This is unusual, but we can handle it.)
                if inputs.shape.rank == 2:
                    inputs = inputs.to_tensor()
                    is_ragged = False
                else:
                    for _ in range(original_inputs.ragged_rank - 1):
                        inputs = inputs.values
                    inputs = inputs.to_tensor()
                    original_inputs = tf.RaggedTensor.from_nested_row_splits(
                        inputs, original_inputs.nested_row_splits[:-1]
                    )

        rank = inputs.shape.rank
        masked_kernel = tf.math.multiply(self.kernel, self.mask, name="masking_kernel")
        if rank == 2 or rank is None:
            # We use embedding_lookup_sparse as a more efficient matmul
            # operation for large sparse input tensors. The op will result in a
            # sparse gradient, as opposed to
            # sparse_ops.sparse_tensor_dense_matmul which results in dense
            # gradients. This can lead to sigfinicant speedups, see b/171762937.
            if isinstance(inputs, tf.SparseTensor):
                # We need to fill empty rows, as the op assumes at least one id
                # per row.
                inputs, _ = tf.sparse.fill_empty_rows(inputs, 0)
                # We need to do some munging of our input to use the embedding
                # lookup as a matrix multiply. We split our input matrix into
                # separate ids and weights tensors. The values of the ids tensor
                # should be the column indices of our input matrix and the
                # values of the weights tensor can continue to the actual matrix
                # weights.  The column arrangement of ids and weights will be
                # summed over and does not matter. See the documentation for
                # sparse_ops.sparse_tensor_dense_matmul a more detailed
                # explanation of the inputs to both ops.
                ids = tf.SparseTensor(
                    indices=inputs.indices,
                    values=inputs.indices[:, 1],
                    dense_shape=inputs.dense_shape,
                )
                weights = inputs
                outputs = tf.nn.embedding_lookup_sparse(
                    masked_kernel, ids, weights, combiner="sum"
                )
            else:
                outputs = tf.matmul(a=inputs, b=masked_kernel)
        # Broadcast kernel to inputs.
        else:
            outputs = tf.tensordot(inputs, masked_kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [masked_kernel.shape[-1]]
                outputs.set_shape(output_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        if is_ragged:
            outputs = original_inputs.with_flat_values(outputs)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "mask": self.mask,
                "activation": keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": keras.regularizers.serialize(
                    self.bias_regularizer
                ),
                "activity_regularizer": keras.regularizers.serialize(
                    self.activity_regularizer
                ),
                "kernel_constraint": keras.constraints.serialize(
                    self.kernel_constraint
                ),
                "bias_constraint": keras.constraints.serialize(self.bias_constraint),
            }
        )
        return config
