import logging
import os
import shutil
import unittest
from mock import patch

import tensorflow as tf

from neural_networks.models import generate_models
from neural_networks.training import train_all_models
from neural_networks.training_mirrored_strategy import train_all_models as train_all_models_mirrored
from utils.setup import SetupNeuralNetworks


class TestCastleSetup(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)

        self.argv = ["-c", "cfg_castle_NN_Creation_test.yml"]

        self.castle_setup = SetupNeuralNetworks(self.argv)

        # Delete existing outputs
        self.nn_output_path = self.castle_setup.nn_output_path
        self.tensorboard_folder = self.castle_setup.tensorboard_folder

        set_memory_growth_gpu()

    def test_create_castle_model_description(self):
        logging.info("Testing creating model descriptions with CASTLE models.")

        self.castle_setup.do_mirrored_strategy = False

        model_descriptions = generate_models(self.castle_setup)

        self.assertIsNotNone(model_descriptions)
        # Check number of models
        # The test config has input: ps (2d) and output: phq (3d)
        # There should be 30 levels for the 3d variable, so we should get 30 models
        self.assertEqual(len(model_descriptions), 30)

    def test_create_castle_model_description_distributed(self):
        logging.info("Testing creating model descriptions with distributed CASTLE models.")

        self.castle_setup.do_mirrored_strategy = True
        if not len(tf.config.list_physical_devices("GPU")):
            logging.warning("Tensorflow found no physical devices. Cannot test distributed strategy without GPUs. "
                            "Exiting test.")
            return

        model_descriptions = generate_models(self.castle_setup)

        self.assertIsNotNone(model_descriptions)
        # Check number of models
        # The test config has input: ps (2d) and output: phq (3d)
        # There should be 30 levels for the 3d variable, so we should get 30 models
        self.assertEqual(len(model_descriptions), 30)

    @patch('neural_networks.models.tf.config.get_visible_devices')
    def test_create_castle_model_description_distributed_value_error(self, mocked_visible_devices):
        logging.info("Testing raise ValueError when creating model descriptions with distributed "
                     "CASTLE models without visible GPUs.")
        # Mock that there aren't any visible devices
        mocked_visible_devices.return_value = []

        self.castle_setup.do_mirrored_strategy = True

        with self.assertRaises(EnvironmentError):
            _ = generate_models(self.castle_setup)


    def test_train_castle_model_description(self):
        logging.info("Testing training 2 model descriptions with CASTLE models.")

        delete_dir(self.tensorboard_folder)
        delete_dir(self.nn_output_path)

        self.castle_setup.do_mirrored_strategy = False

        model_descriptions = generate_models(self.castle_setup)

        # Only test train the first two models
        train_model_descriptions = model_descriptions[:2]
        train_all_models(train_model_descriptions, self.castle_setup)

        self._assert_saved_files(train_model_descriptions)

    def test_train_castle_model_description_distributed(self):
        logging.info("Testing distributed training of 2 model descriptions with distributed CASTLE models.")

        delete_dir(self.tensorboard_folder)
        delete_dir(self.nn_output_path)

        self.castle_setup.do_mirrored_strategy = True
        if not len(tf.config.list_physical_devices("GPU")):
            logging.warning("Tensorflow found no physical devices. Cannot test distributed strategy without GPUs. "
                            "Exiting test.")
            return

        model_descriptions = generate_models(self.castle_setup)

        # Only test train the first two models
        train_model_descriptions = model_descriptions[:2]
        train_all_models_mirrored(train_model_descriptions, self.castle_setup)

        self._assert_saved_files(train_model_descriptions)

    def _assert_saved_files(self, train_model_descriptions):
        for m in train_model_descriptions:
            model_fn = m.get_filename() + '_model.h5'
            out_path = str(m.get_path(self.castle_setup.nn_output_path))

            self.assertTrue(os.path.isfile(os.path.join(out_path, model_fn)))
            self.assertTrue(os.path.isdir(self.tensorboard_folder))


def delete_dir(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)


def set_memory_growth_gpu():
    physical_devices = tf.config.list_physical_devices("GPU")
    print(f"Number of GPUs: {len(physical_devices)}", flush=True)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
