import logging
import unittest

import tensorflow as tf
from tensorflow import keras

from neural_networks.castle import build_castle


class TestCastle(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)

        self.num_inputs = 10
        self.hidden_layers = [5, 5, 5]
        self.leaky_relu = "leakyReLU"
        self.relu = "relu"
        self.rho = 1.0
        self.alpha = 1.0
        self.lambda_ = 1.0

    def test_castle_model(self):
        logging.info("Testing building and compiling CASTLE model.")

        model = build_castle(self.num_inputs, self.hidden_layers, self.leaky_relu, self.rho, self.alpha, self.lambda_,
                             eager_execution=True, seed=42)
        self.assertIsNotNone(model)

        print(model.summary())
        try:
            keras.utils.plot_model(model, to_file="castle.png", show_shapes=True, show_layer_activations=True)
        except ImportError:
            print("WARNING: Cannot plot model because either pydot or graphviz are not installed. "
                  "See tf.keras.utils.plot_model documentation for details.")

    def test_castle_model_mirrored_strategy(self):
        logging.info("Testing building and compiling CASTLE model with tf.distribute.MirroredStrategy.")

        strategy = tf.distribute.MirroredStrategy()

        model = build_castle(self.num_inputs, self.hidden_layers, self.relu, self.rho, self.alpha, self.lambda_,
                             eager_execution=True, strategy=strategy, seed=42)
        self.assertIsNotNone(model)

        print(model.summary())
        try:
            keras.utils.plot_model(model, to_file="castle.png", show_shapes=True, show_layer_activations=True)
        except ImportError:
            print("WARNING: Cannot plot model because either pydot or graphviz are not installed. "
                  "See tf.keras.utils.plot_model documentation for details.")
