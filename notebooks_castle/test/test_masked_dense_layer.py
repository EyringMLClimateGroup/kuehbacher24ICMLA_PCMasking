from neural_networks.castle.masked_dense_layer import MaskedDenseLayer
import unittest
import numpy as np
from notebooks_castle.test.testing_utils import set_memory_growth_gpu


class TestMaskedDenseLayer(unittest.TestCase):

    def setUp(self):
        self.units = 10
        self.shape = (32, self.units)
        self.mask = np.ones(self.shape)
        masked_column = 2
        self.mask[masked_column, :] = 0

        try:
            set_memory_growth_gpu()
        except RuntimeError:
            print("\nGPU growth could not be enabled. "
                  "When running multiple tests, this may be because the physical drivers are already "
                  "initialized, in which case memory growth may already be enabled. "
                  "If memory growth is not enabled, the tests may fail with CUDA error.\n")

    def test_create_masked_dense_layer(self):
        _ = MaskedDenseLayer(self.units, self.mask)


if __name__ == "__main__":
    unittest.main()
