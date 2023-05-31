import unittest

from tensorflow import keras

from neural_networks.castle import build_castle


class TestCastle(unittest.TestCase):

    def test_castle_model(self):
        num_inputs = 10
        hidden_layers = [5, 5, 5]
        activation = "leakyReLU"
        rho = 1.0
        alpha = 1.0
        lambda_ = 1.0

        model = build_castle(num_inputs, hidden_layers, activation, rho, alpha, lambda_)
        print(model.summary())
        try:
            keras.utils.plot_model(model, to_file="castle.png", show_shapes=True, show_layer_activations=True)
        except ImportError:
            print("WARNING: Cannot plot model because either pydot or graphviz are not installed. "
                  "See tf.keras.utils.plot_model documentation for details.")
