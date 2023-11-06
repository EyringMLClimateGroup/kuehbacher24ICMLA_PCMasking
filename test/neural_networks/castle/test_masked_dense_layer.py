from neural_networks.castle.masked_dense_layer import MaskedDenseLayer
import pytest
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
    units = 10
    shape = (32, units)

    mask = np.ones(shape)
    masked_column = 2
    mask[masked_column, :] = 0

    mdl = MaskedDenseLayer(units, mask)

    assert (isinstance(mdl, MaskedDenseLayer))
    assert (units == mdl.units)
    assert (np.alltrue(mdl.mask == mask))
