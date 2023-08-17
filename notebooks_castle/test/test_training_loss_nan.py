import logging
import os
import unittest

import tensorflow as tf

from neural_networks.models import generate_models
from neural_networks.training import train_all_models
from neural_networks.training_mirrored_strategy import train_all_models as train_all_models_mirrored
from notebooks_castle.test.testing_utils import delete_dir, set_memory_growth_gpu
from utils.setup import SetupNeuralNetworks


# The purpose of these tests is to see whether the training loss is still a number,
# when we have a full network with 94 inputs and 1 outputs. They are separate from normal training tests
# in order to speed up those tests.
# Unfortunately, we don't have access to the training history, so we need to monitor the output.
#
# Possible options for reducing the training loss:
#  - reduce initial weights
#  - reduce learning rate
#  - adam options: weight decay, clip norm, clip value
class TestTrainingLossNaN(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)

        self.argv = ["-c", "config/cfg_castle_NN_Creation_test_2.yml"]

        self.castle_setup = SetupNeuralNetworks(self.argv)

        # Delete existing outputs
        self.nn_output_path = self.castle_setup.nn_output_path
        self.tensorboard_folder = self.castle_setup.tensorboard_folder

        try:
            set_memory_growth_gpu()
        except RuntimeError:
            logging.warning("GPU growth could not be enabled. "
                            "When running multiple tests, this may be because the physical drivers are already "
                            "initialized, in which case memory growth may already be enabled. "
                            "If memory growth is not enabled, the tests may fail with CUDA error.")

    def test_train_castle_model_description(self):
        logging.info("Testing training loss float overflow (NaN).")

        delete_dir(self.tensorboard_folder)
        delete_dir(self.nn_output_path)

        self.castle_setup.distribute_strategy = ""

        model_descriptions = generate_models(self.castle_setup)

        # Only test train the first two models
        train_model_descriptions = model_descriptions[:1]
        train_all_models(train_model_descriptions, self.castle_setup)
        # Assert: Monitor output that training loss is not NaN

    def test_train_castle_model_description_distributed(self):
        logging.info("Testing training loss float overflow (NaN) in distributed setting.")

        delete_dir(self.tensorboard_folder)
        delete_dir(self.nn_output_path)

        self.castle_setup.distribute_strategy = "mirrored"
        if not len(tf.config.list_physical_devices("GPU")):
            logging.warning("Tensorflow found no physical devices. Cannot test distributed strategy without GPUs. "
                            "Exiting test.")
            return

        model_descriptions = generate_models(self.castle_setup)

        # Only test train the first two models
        train_model_descriptions = model_descriptions[:1]
        train_all_models_mirrored(train_model_descriptions, self.castle_setup)
        # Assert: Monitor the output that training loss is not NaN
