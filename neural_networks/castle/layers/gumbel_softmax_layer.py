import tensorflow as tf
from tensorflow_probability.python.distributions import RelaxedOneHotCategorical

from neural_networks.castle.castle_model_base import get_kernel_initializer


@tf.keras.utils.register_keras_serializable()
class StraightThroughGumbelSoftmaxMaskingLayer(tf.keras.layers.Layer):
    def __init__(self, num_vars, temp=1.0,
                 params_initializer='glorot_uniform',
                 params_regularizer=None,
                 params_constraint=None, seed=None,
                 name="straight_through_gumbel_softmax_masking_layer", **kwargs):
        super(StraightThroughGumbelSoftmaxMaskingLayer, self).__init__(name=name, **kwargs)

        self.num_vars = num_vars

        self.temp = tf.Variable(temp, trainable=False, name="temperature")

        self.params_initializer = params_initializer
        self.params_regularizer = params_regularizer
        self.params_constraint = params_constraint

        self.seed = seed

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])

        if last_dim != self.num_vars:
            raise ValueError(f"Last input dimensions does not match number of variables: {last_dim} != {self.num_vars}")

        self.params_vector = self.add_weight("params_vector",
                                             shape=[last_dim, ],
                                             initializer=get_kernel_initializer(self.params_initializer, self.seed),
                                             regularizer=self.params_regularizer,
                                             constraint=self.params_constraint,
                                             trainable=True)

        self.masking_vector = tf.Variable(tf.ones_like(self.params_vector, dtype=self.params_vector.dtype),
                                          trainable=False, name="masking_vector")

    def call(self, inputs, training=None):
        self.masking_vector.assign(self.sample_masking_vector(self.params_vector, self.temp))

        masked_inputs = tf.math.multiply(inputs, self.masking_vector)

        return masked_inputs

    @staticmethod
    def sample_masking_vector(params, temp):
        # Create a tensor of zeros with the same shape as params to represent the fixed logits for class 0
        zeros = tf.zeros_like(params)

        # Stack the params and zeros along a new dimension to create logits
        # The resulting shape will be [num_vars, 2], where the last dimension represents logits for two classes
        logits = tf.stack([zeros, params], axis=-1)

        # Apply Gumbel-Softmax to the logits
        gumbel_softmax_sample = RelaxedOneHotCategorical(temp, logits=logits).sample()

        # Select the class 1 probabilities
        binary_mask = gumbel_softmax_sample[:, 1]

        return binary_mask

    def get_config(self):
        config = super(StraightThroughGumbelSoftmaxMaskingLayer, self).get_config()
        config.update({
            "num_vars": self.num_vars,
            "temp": self.temp,
            "params_initializer": self.params_initializer,
            "params_regularizer": self.params_regularizer,
            "params_constraint": self.params_constraint,
        })
        return config
