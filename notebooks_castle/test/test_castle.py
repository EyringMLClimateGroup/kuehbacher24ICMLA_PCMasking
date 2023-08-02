import logging
import os
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from neural_networks.castle_model import CASTLE
from neural_networks.castle import build_castle
from notebooks_castle.test.testing_utils import set_memory_growth_gpu


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

        self.output_dir = os.path.join(Path(__file__).parent.resolve(), "output")

        try:
            set_memory_growth_gpu()
        except RuntimeError:
            logging.warning("GPU growth could not be enabled. "
                            "When running multiple tests, this may be because the physical drivers are already "
                            "initialized, in which case memory growth may already be enabled. "
                            "If memory growth is not enabled, the tests may fail with CUDA error.")

    def test_castle_model(self):
        logging.info("Testing building and compiling CASTLE model.")

        model = build_castle(self.num_inputs, self.hidden_layers, self.leaky_relu, self.rho, self.alpha, self.lambda_,
                             eager_execution=True, seed=42)
        self.assertIsNotNone(model)

        print(model.summary())
        try:
            keras.utils.plot_model(model, to_file=Path(self.output_dir, "castle.png"), show_shapes=True,
                                   show_layer_activations=True)
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

    def test_train_castle_model(self):
        logging.info("Testing training CASTLE model.")

        model = build_castle(self.num_inputs, self.hidden_layers, self.leaky_relu, self.rho, self.alpha, self.lambda_,
                             eager_execution=True, seed=42)

        epochs = 3
        history = self.train_castle(model, epochs=epochs)

        self.assertIsNotNone(history)

        train_loss_keys = ["loss", "prediction_loss", "reconstruction_loss", "sparsity_loss", "acyclicity_loss"]
        val_loss_keys = ["val_" + loss for loss in train_loss_keys]
        self.assertTrue(all(k in history.history.keys() for k in train_loss_keys))
        self.assertTrue(all(k in history.history.keys() for k in val_loss_keys))

        self.assertEqual(len(history.history["loss"]), epochs)

    def test_predict_castle_model(self):
        logging.info("Testing predicting with CASTLE model.")

        model = build_castle(self.num_inputs, self.hidden_layers, self.leaky_relu, self.rho, self.alpha, self.lambda_,
                             eager_execution=True, seed=42)

        n_samples = 320
        batch_size = 32
        num_batches = int(n_samples / batch_size)
        num_outputs = 1

        x_array = np.random.rand(n_samples, self.num_inputs)
        y_array = np.zeros((n_samples, num_outputs), dtype=np.float32)

        test_ds = tf.data.Dataset.from_tensor_slices((x_array, y_array)).batch(batch_size, drop_remainder=True)

        prediction = model.predict(test_ds)

        self.assertIsNotNone(prediction)
        self.assertEqual((batch_size * num_batches, self.num_inputs + 1), prediction.shape)

    def test_save_load_castle_model(self):
        model = build_castle(self.num_inputs, self.hidden_layers, self.leaky_relu, self.rho, self.alpha, self.lambda_,
                             eager_execution=True, seed=42)

        _ = self.train_castle(model, epochs=1)

        model.save(Path(self.output_dir, "castle_model.keras"), save_format="keras_v3")
        model.save_weights(str(Path(self.output_dir, "castle_weights.h5")))

        loaded_model = tf.keras.models.load_model(Path(self.output_dir, "castle_model.keras"),
                                                  custom_objects={'CASTLE': CASTLE})

        self.assertEqual(loaded_model.alpha, model.alpha)
        self.assertEqual(loaded_model.rho, model.rho)
        self.assertEqual(loaded_model.reg_lambda, model.reg_lambda)
        self.assertEqual(len(loaded_model.get_weights()), len(model.get_weights()))

    def train_castle(self, model, epochs):
        n_samples = 320
        num_outputs = 1
        batch_size = 32

        x_array = np.random.standard_normal((n_samples, self.num_inputs)).astype(dtype=np.float32)
        y_array = np.random.standard_normal((n_samples, num_outputs)).astype(dtype=np.float32)
        train_ds = tf.data.Dataset.from_tensor_slices((x_array, y_array)).batch(batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((x_array, y_array)).batch(batch_size)

        history = model.fit(
            x=train_ds,
            validation_data=val_ds,
            batch_size=batch_size,
            epochs=epochs
        )

        return history
