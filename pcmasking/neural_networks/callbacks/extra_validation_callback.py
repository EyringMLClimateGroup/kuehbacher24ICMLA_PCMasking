"""
Code from: https://github.com/tanzhenyu/image_augmentation/blob/master/image_augmentation/callbacks/extra_eval.py
Apache License 2.0
Altered content
"""
import tensorflow as tf
from tensorflow import keras
from keras.utils import io_utils
import os


class ExtraValidation(keras.callbacks.Callback):
    """
    Callback for logging extra validation datasets. This functionality is useful for model training scenarios
    where validation on multiple validation is desirable (Keras by default, provides functionality for
    evaluating on a single validation set only).

    The evaluation metrics are printed during training but are only logged to Tensorboard,
    if this callback is added to the list of callbacks passed to the model in
    ``model.fit`` **before** the Tensorboard callback.

    Args:
        validation_sets (dict): Dictionary of the form {"name": dataset_name, "data": dataset}, where
            the dataset is an extra validation dataset.
        tensorboard_path: Path to the TensorBoard logging directory.
    """

    def __init__(self, validation_sets, tensorboard_path, log_iterations=True):
        super(ExtraValidation, self).__init__()

        self.validation_sets = validation_sets

        self.tensorboard_path = io_utils.path_to_string(tensorboard_path)
        self._val_dir = os.path.join(self.tensorboard_path, "validation")

        self.log_iterations = log_iterations
        if self.log_iterations:
            self._val_writer = tf.summary.create_file_writer(self.tensorboard_path)

    def on_epoch_end(self, batch, logs=None):
        additional_logs = {}
        for dataset_name, dataset in self.validation_sets.items():
            scores = self.model.evaluate(dataset, verbose=2)
            # If the model has multiple metrics, scores is already a list.
            # If there is a single or no metric, we convert it to list
            if not isinstance(scores, list):
                scores = [scores]

            for metric, score in zip(self.model.metrics, scores):
                additional_logs[f"val_{dataset_name}_{metric.name}"] = score
        logs.update(additional_logs)

        if self.log_iterations:
            if self.model.optimizer and hasattr(self.model.optimizer, "iterations"):
                with tf.summary.record_if(True), self._val_writer.as_default():
                    for name, value in additional_logs.items():
                        name = name[4:]  # Remove 'val_' prefix.
                        tf.summary.scalar(
                            "evaluation_" + name + "_vs_iterations", value,
                            step=self.model.optimizer.iterations.read_value(),
                        )
