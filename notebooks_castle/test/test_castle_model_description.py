import logging
import os
import unittest

import tensorflow as tf
from mock import patch

from neural_networks.load_models import load_models, load_model_weights_from_checkpoint, \
    load_model_from_previous_training
from neural_networks.models import generate_models
from neural_networks.training import train_all_models
from neural_networks.training_mirrored_strategy import train_all_models as train_all_models_mirrored
from notebooks_castle.test.testing_utils import delete_output_dirs, set_memory_growth_gpu, train_model_if_not_exists, \
    build_test_gen
from utils.setup import SetupNeuralNetworks


class TestCastleModelDescription(unittest.TestCase):

    def setUp(self):

        try:
            set_memory_growth_gpu()
        except RuntimeError:
            print("\nGPU growth could not be enabled. "
                  "When running multiple tests, this may be because the physical drivers are already "
                  "initialized, in which case memory growth may already be enabled. "
                  "If memory growth is not enabled, the tests may fail with CUDA error.")

        # Multiple inputs and outputs, both 2d and 3d
        # Inputs: ps (3d) and lhflx (2d)
        # Outputs: phq (3d) and prect (2d)
        #   There should be 30 levels for the 3d variable, so we should get 31 single output networks
        argv_s1 = ["-c", "config/cfg_castle_NN_Creation_test_1.yml"]
        self.num_models_config_1 = 31
        # Just two 2d variables, which results in 2 single output networks
        argv_s2 = ["-c", "config/cfg_castle_NN_Creation_test_2.yml"]

        self.castle_setup_many_networks = SetupNeuralNetworks(argv_s1)
        self.castle_setup_few_networks = SetupNeuralNetworks(argv_s2)

    def test_create_castle_model_description(self):
        print("\nTesting creating model description instances with CASTLE models.")

        self.castle_setup_many_networks.distribute_strategy = ""

        model_descriptions = generate_models(self.castle_setup_many_networks)

        self.assertIsNotNone(model_descriptions)
        # Check number of models
        self.assertEqual(len(model_descriptions), self.num_models_config_1)

    def test_create_castle_model_description_distributed(self):
        print("\nTesting creating model description instances with distributed CASTLE models.")

        self.castle_setup_many_networks.distribute_strategy = "mirrored"
        if not len(tf.config.list_physical_devices("GPU")):
            print("\nTensorflow found no physical devices. Cannot test distributed strategy without GPUs. "
                  "Exiting test.")
            return

        model_descriptions = generate_models(self.castle_setup_many_networks)

        self.assertIsNotNone(model_descriptions)
        # Check number of models
        self.assertEqual(len(model_descriptions), self.num_models_config_1)

    @patch('neural_networks.models.tf.config.get_visible_devices')
    def test_create_castle_model_description_distributed_value_error(self, mocked_visible_devices):
        print("\nTesting raise ValueError when creating model description instances with distributed "
              "CASTLE models without visible GPUs.")
        # Mock that there aren't any visible devices
        mocked_visible_devices.return_value = []

        self.castle_setup_many_networks.distribute_strategy = "mirrored"

        with self.assertRaises(EnvironmentError):
            _ = generate_models(self.castle_setup_many_networks)

    def test_train_and_save_castle_model_description(self):
        print("\nTesting training 2 model description instances with CASTLE models.")

        self.castle_setup_many_networks.distribute_strategy = ""

        model_descriptions = generate_models(self.castle_setup_many_networks)
        delete_output_dirs(model_descriptions, self.castle_setup_many_networks)

        # Only test train the first two models
        train_model_descriptions = model_descriptions[:2]
        train_all_models(train_model_descriptions, self.castle_setup_many_networks)

        self._assert_saved_files(train_model_descriptions)

    def test_train_and_save_castle_model_description_distributed(self):
        print("\nTesting distributed training of 2 model description instances with distributed CASTLE models.")

        self.castle_setup_many_networks.distribute_strategy = "mirrored"

        if not len(tf.config.list_physical_devices("GPU")):
            print("\nTensorflow found no physical devices. Cannot test distributed strategy without GPUs. "
                  "Exiting test.")
            return

        model_descriptions = generate_models(self.castle_setup_many_networks)
        delete_output_dirs(model_descriptions, self.castle_setup_many_networks)

        # Only test train the first two models
        train_model_descriptions = model_descriptions[:2]
        train_all_models_mirrored(train_model_descriptions, self.castle_setup_many_networks)

        self._assert_saved_files(train_model_descriptions)

    def _assert_saved_files(self, train_model_descriptions):
        for m in train_model_descriptions:
            model_fn = m.get_filename() + '_model.keras'
            out_path = str(m.get_path(self.castle_setup_many_networks.nn_output_path))

            self.assertTrue(os.path.isfile(os.path.join(out_path, model_fn)))
            self.assertTrue(os.path.isdir(self.castle_setup_many_networks.tensorboard_folder))

    def test_load_castle_model_description(self):
        print("\nTesting loading of model description instance with trained CASTLE models.")

        self.castle_setup_few_networks.distribute_strategy = ""

        train_model_if_not_exists(self.castle_setup_few_networks)
        loaded_model_description = load_models(self.castle_setup_few_networks)

        self.assertEqual(len(loaded_model_description[self.castle_setup_few_networks.nn_type]),
                         len(self.castle_setup_few_networks.output_order))

    def test_load_castle_model_description_distributed(self):
        print("\nTesting loading of model description instance with distributed trained CASTLE models.")

        train_model_if_not_exists(self.castle_setup_few_networks)

        loaded_model_description = load_models(self.castle_setup_few_networks)

        self.assertEqual(len(loaded_model_description[self.castle_setup_few_networks.nn_type]),
                         len(self.castle_setup_few_networks.output_order))
