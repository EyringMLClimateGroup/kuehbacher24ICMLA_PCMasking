import logging
import os
import unittest

import tensorflow as tf
from mock import patch

from neural_networks.load_models import load_models
from neural_networks.models import generate_models
from neural_networks.training import train_all_models
from neural_networks.training_mirrored_strategy import train_all_models as train_all_models_mirrored
from notebooks_castle.test.testing_utils import delete_dir, set_memory_growth_gpu
from utils.setup import SetupNeuralNetworks


class TestCastleSetup(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)

        try:
            set_memory_growth_gpu()
        except RuntimeError:
            logging.warning("GPU growth could not be enabled. "
                            "When running multiple tests, this may be because the physical drivers are already "
                            "initialized, in which case memory growth may already be enabled. "
                            "If memory growth is not enabled, the tests may fail with CUDA error.")

        # Multiple inputs and outputs, both 2d and 3d
        # Inputs: ps (3d) and lhflx (2d)
        # Outputs: phq (3d) and prect (2d)
        #   There should be 30 levels for the 3d variable, so we should get 31 single output networks
        argv_s1 = ["-c", "cfg_castle_NN_Creation_test_1.yml"]
        self.num_models_config_1 = 31
        # Just two 2d variables, which results in 2 single output networks
        argv_s2 = ["-c", "cfg_castle_NN_Creation_test_2.yml"]

        self.castle_setup_many_networks = SetupNeuralNetworks(argv_s1)
        self.castle_setup_few_networks = SetupNeuralNetworks(argv_s2)

    def test_create_castle_model_description(self):
        logging.info("Testing creating model description instances with CASTLE models.")

        self.castle_setup_many_networks.do_mirrored_strategy = False

        model_descriptions = generate_models(self.castle_setup_many_networks)

        self.assertIsNotNone(model_descriptions)
        # Check number of models
        self.assertEqual(len(model_descriptions), self.num_models_config_1)

    def test_create_castle_model_description_distributed(self):
        logging.info("Testing creating model description instances with distributed CASTLE models.")

        self.castle_setup_many_networks.do_mirrored_strategy = True
        if not len(tf.config.list_physical_devices("GPU")):
            logging.warning("Tensorflow found no physical devices. Cannot test distributed strategy without GPUs. "
                            "Exiting test.")
            return

        model_descriptions = generate_models(self.castle_setup_many_networks)

        self.assertIsNotNone(model_descriptions)
        # Check number of models
        self.assertEqual(len(model_descriptions), self.num_models_config_1)

    @patch('neural_networks.models.tf.config.get_visible_devices')
    def test_create_castle_model_description_distributed_value_error(self, mocked_visible_devices):
        logging.info("Testing raise ValueError when creating model description instances with distributed "
                     "CASTLE models without visible GPUs.")
        # Mock that there aren't any visible devices
        mocked_visible_devices.return_value = []

        self.castle_setup_many_networks.do_mirrored_strategy = True

        with self.assertRaises(EnvironmentError):
            _ = generate_models(self.castle_setup_many_networks)

    def test_train_and_save_castle_model_description(self):
        logging.info("Testing training 2 model description instances with CASTLE models.")

        # Delete existing output directories
        delete_dir(self.castle_setup_many_networks.nn_output_path)
        delete_dir(self.castle_setup_many_networks.tensorboard_folder)

        self.castle_setup_many_networks.do_mirrored_strategy = False

        model_descriptions = generate_models(self.castle_setup_many_networks)

        # Only test train the first two models
        train_model_descriptions = model_descriptions[:2]
        train_all_models(train_model_descriptions, self.castle_setup_many_networks)

        self._assert_saved_files(train_model_descriptions)

    def test_train_and_save_castle_model_description_distributed(self):
        logging.info("Testing distributed training of 2 model description instances with distributed CASTLE models.")

        # Delete existing output directories
        delete_dir(self.castle_setup_many_networks.nn_output_path)
        delete_dir(self.castle_setup_many_networks.tensorboard_folder)

        self.castle_setup_many_networks.do_mirrored_strategy = True
        if not len(tf.config.list_physical_devices("GPU")):
            logging.warning("Tensorflow found no physical devices. Cannot test distributed strategy without GPUs. "
                            "Exiting test.")
            return

        model_descriptions = generate_models(self.castle_setup_many_networks)

        # Only test train the first two models
        train_model_descriptions = model_descriptions[:2]
        train_all_models_mirrored(train_model_descriptions, self.castle_setup_many_networks)

        self._assert_saved_files(train_model_descriptions)

    def _assert_saved_files(self, train_model_descriptions):
        for m in train_model_descriptions:
            model_fn = m.get_filename() + '_model.h5'
            out_path = str(m.get_path(self.castle_setup_many_networks.nn_output_path))

            self.assertTrue(os.path.isfile(os.path.join(out_path, model_fn)))
            self.assertTrue(os.path.isdir(self.castle_setup_many_networks.tensorboard_folder))

    def test_load_castle_model_description(self):
        logging.info("Testing loading of model description instance with trained CASTLE models.")

        self.castle_setup_few_networks.do_mirrored_strategy = False
        model_descriptions = generate_models(self.castle_setup_few_networks)

        for md in model_descriptions:
            trained_model = md.get_filename() + '_model.h5'
            training_path = str(md.get_path(self.castle_setup_few_networks.nn_output_path))
            if not os.path.isfile(os.path.join(training_path, trained_model)):
                train_all_models([md], self.castle_setup_few_networks)

        loaded_model_description = load_models(self.castle_setup_few_networks)

        self.assertEqual(len(loaded_model_description[self.castle_setup_few_networks.nn_type]),
                         len(self.castle_setup_few_networks.output_order))

    def test_load_castle_model_description_distributed(self):
        logging.info("Testing loading of model description instance with distributed trained CASTLE models.")

        self.castle_setup_few_networks.do_mirrored_strategy = False
        model_descriptions = generate_models(self.castle_setup_few_networks)

        for md in model_descriptions:
            trained_model = md.get_filename() + '_model.h5'
            training_path = str(md.get_path(self.castle_setup_few_networks.nn_output_path))
            if not os.path.isfile(os.path.join(training_path, trained_model)):
                train_all_models_mirrored([md], self.castle_setup_few_networks)

        loaded_model_description = load_models(self.castle_setup_few_networks)

        self.assertEqual(len(loaded_model_description[self.castle_setup_few_networks.nn_type]),
                         len(self.castle_setup_few_networks.output_order))
