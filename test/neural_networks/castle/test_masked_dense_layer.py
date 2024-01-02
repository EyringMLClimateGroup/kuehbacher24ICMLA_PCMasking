from neural_networks.castle.layers.masked_dense_layer import MaskedDenseLayer
import numpy as np
from test.testing_utils import set_memory_growth_gpu

try:
    set_memory_growth_gpu()
except RuntimeError:
    print("\n\n*** GPU growth could not be enabled. "
          "When running multiple tests, this may be due physical drivers having already been "
          "initialized, in which case memory growth may already be enabled. "
          "If memory growth is not enabled, the tests may fail with CUDA error. ***\n")


def test_create_masked_dense_layer():
    input_shape = 10
    units = 16
    shape = (input_shape, units)

    mask = np.ones(shape, dtype=np.float32)
    masked_row = 2
    mask[:, masked_row] = 0

    mdl = MaskedDenseLayer(units, mask)
    mdl.build(input_shape=(None, input_shape))

    assert (isinstance(mdl, MaskedDenseLayer))
    assert (units == mdl.units)
    assert (np.alltrue(mdl.mask == mask))


def test_masked_dense_layer_masking():
    input_shape = 10
    units = 16
    shape = (input_shape, units)

    mask = np.ones(shape, dtype=np.float32)
    masked_row = 2
    mask[:, masked_row] = 0

    mdl = MaskedDenseLayer(units, mask, kernel_initializer="ones")

    input = np.ones((32, input_shape), dtype=np.float32)
    output = mdl(input)

    print(output)

    assert (all(output[:, masked_row] == 0))

