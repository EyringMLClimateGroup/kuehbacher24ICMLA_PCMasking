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


class TestLoadTrainCastleModelDescription(unittest.TestCase):

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

    def test_load_ckpt_castle_model_description(self):
        print("\nTesting loading model weights from checkpoint for CASTLE models.")

        self.castle_setup_few_networks.distribute_strategy = ""

        train_model_if_not_exists(self.castle_setup_few_networks)
        model_descriptions = generate_models(self.castle_setup_few_networks)

        for md in model_descriptions:
            # Evaluate the model
            test_gen = build_test_gen(md, self.castle_setup_few_networks)
            print(f"\nEvaluated untrained model {md}.")
            with test_gen:
                md.model.evaluate(test_gen, verbose=2)

            md = load_model_weights_from_checkpoint(md, which_checkpoint="best")

            print(f"\nEvaluated model {md} with loaded weights.")
            with test_gen:
                md.model.evaluate(test_gen, verbose=2)

    def test_load_ckpt_castle_model_description_distributed(self):
        print("\nTesting loading model weights from checkpoint for CASTLE models.")

        self.castle_setup_few_networks.distribute_strategy = "mirrored"

        train_model_if_not_exists(self.castle_setup_few_networks)
        model_descriptions = generate_models(self.castle_setup_few_networks)

        for md in model_descriptions:
            # Evaluate the model
            test_gen = build_test_gen(md, self.castle_setup_few_networks)
            print(f"\nEvaluated untrained model {md}.")
            with test_gen:
                md.model.evaluate(test_gen, verbose=2)

            md = load_model_weights_from_checkpoint(md, which_checkpoint="best")

            print(f"\nEvaluated model {md} with loaded weights.")
            with test_gen:
                md.model.evaluate(test_gen, verbose=2)

    def test_train_load_ckpt_castle_model_description(self):
        print("\nTesting continue training with loaded model weights for CASTLE model.")

        self.castle_setup_few_networks.distribute_strategy = ""

        model_descriptions = generate_models(self.castle_setup_few_networks)
        delete_output_dirs(model_descriptions, self.castle_setup_few_networks)

        # First training
        train_all_models(model_descriptions, self.castle_setup_few_networks)

        del model_descriptions

        # Train again from checkpoint
        model_descriptions = generate_models(self.castle_setup_few_networks)
        train_all_models(model_descriptions, self.castle_setup_few_networks, from_checkpoint=True)

    def test_train_load_ckpt_castle_model_description_distributed(self):
        print("\nTesting continue training with loaded model weights for CASTLE model.")

        self.castle_setup_few_networks.distribute_strategy = "mirrored"

        model_descriptions = generate_models(self.castle_setup_few_networks)
        delete_output_dirs(model_descriptions, self.castle_setup_few_networks)

        # First training
        train_all_models(model_descriptions, self.castle_setup_few_networks)

        del model_descriptions

        # Train again from checkpoint
        model_descriptions = generate_models(self.castle_setup_few_networks)
        train_all_models(model_descriptions, self.castle_setup_few_networks, from_checkpoint=True)

    def test_load_whole_castle_model_description(self):
        print("\nTesting loading the whole model from previous training for CASTLE models.")

        self.castle_setup_few_networks.distribute_strategy = ""

        train_model_if_not_exists(self.castle_setup_few_networks)
        model_descriptions = generate_models(self.castle_setup_few_networks)

        for md in model_descriptions:
            # Evaluate the model
            test_gen = build_test_gen(md, self.castle_setup_few_networks)
            print(f"\nEvaluated untrained model {md}.")
            with test_gen:
                md.model.evaluate(test_gen, verbose=2)

            md.model = load_model_from_previous_training(md)

            print(f"\nEvaluated model {md} with loaded weights.")
            with test_gen:
                md.model.evaluate(test_gen, verbose=2)

    def test_load_whole_castle_model_description_distributed(self):
        print("\nTesting loading whole model from previous training for CASTLE models.")

        self.castle_setup_few_networks.distribute_strategy = "mirrored"

        train_model_if_not_exists(self.castle_setup_few_networks)
        model_descriptions = generate_models(self.castle_setup_few_networks)

        for md in model_descriptions:
            # Evaluate the model
            test_gen = build_test_gen(md, self.castle_setup_few_networks)
            print(f"\nEvaluated untrained model {md}.")
            with test_gen:
                md.model.evaluate(test_gen, verbose=2)

            md.model = load_model_from_previous_training(md)

            print(f"\nEvaluated model {md} with loaded weights.")
            with test_gen:
                md.model.evaluate(test_gen, verbose=2)

    def test_train_load_whole_castle_model_description(self):
        print("\nTesting continue training with loaded model for CASTLE model.")

        self.castle_setup_few_networks.distribute_strategy = ""

        model_descriptions = generate_models(self.castle_setup_few_networks)
        delete_output_dirs(model_descriptions, self.castle_setup_few_networks)

        # First training
        train_all_models(model_descriptions, self.castle_setup_few_networks)

        del model_descriptions

        # Train again from checkpoint
        model_descriptions = generate_models(self.castle_setup_few_networks, continue_training=True)
        train_all_models(model_descriptions, self.castle_setup_few_networks, continue_training=True)

    def test_train_load_whole_castle_model_description_distributed(self):
        print("\nTesting continue training with loaded model for CASTLE model.")

        self.castle_setup_few_networks.distribute_strategy = "mirrored"

        model_descriptions = generate_models(self.castle_setup_few_networks)
        delete_output_dirs(model_descriptions, self.castle_setup_few_networks)

        # First training
        train_all_models(model_descriptions, self.castle_setup_few_networks)

        del model_descriptions

        # Train again from checkpoint
        model_descriptions = generate_models(self.castle_setup_few_networks, continue_training=True)
        train_all_models(model_descriptions, self.castle_setup_few_networks, continue_training=True)
