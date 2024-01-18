import pytest

from neural_networks.castle.layers.gumbel_softmax_layer import StraightThroughGumbelSoftmaxMaskingLayer
import numpy as np
from test.testing_utils import set_memory_growth_gpu

try:
    set_memory_growth_gpu()
except RuntimeError:
    print("\n\n*** GPU growth could not be enabled. "
          "When running multiple tests, this may be due physical drivers having already been "
          "initialized, in which case memory growth may already be enabled. "
          "If memory growth is not enabled, the tests may fail with CUDA error. ***\n")


def test_create_gumbel_softmax_masking_layer():
    num_vars = 5
    temp = 2.0

    masking_layer = StraightThroughGumbelSoftmaxMaskingLayer(num_vars, temp=temp)

    with pytest.raises(AttributeError):
        masking_layer.masking_vector

    masking_layer.build(input_shape=(None, num_vars))

    assert (isinstance(masking_layer, StraightThroughGumbelSoftmaxMaskingLayer))
    assert (num_vars == masking_layer.num_vars)
    assert (temp == masking_layer.temp)

    assert (len(masking_layer.trainable_variables) == 1)

    assert (masking_layer.trainable_variables[0].shape[-1] == num_vars)
    assert (masking_layer.masking_vector is not None)
    assert (masking_layer.masking_vector.shape == masking_layer.params_vector.shape)


def test_gumbel_softmax_layer_masking():
    num_classes = 8
    test_data = np.random.randn(num_classes)

    masking_layer = StraightThroughGumbelSoftmaxMaskingLayer(num_classes)
    masking_layer.build(input_shape=(num_classes,))

    masking_layer(test_data)

    assert (masking_layer.masking_vector is not None)
    assert (masking_layer.masking_vector.shape == masking_layer.params_vector.shape)
