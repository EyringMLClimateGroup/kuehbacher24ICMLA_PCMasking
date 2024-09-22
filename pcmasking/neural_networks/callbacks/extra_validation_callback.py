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
    Custom Keras callback to log evaluation metrics for additional validation datasets.

    This callback allows evaluation on multiple validation datasets during the training
    process. It complements the default Keras functionality which only supports one validation
    dataset by logging additional metrics to TensorBoard. This is particularly useful when
    you have several validation datasets that represent different data distributions or
    task-specific subsets.

    The additional evaluation metrics will be logged and printed during training. To ensure
    they are properly recorded in TensorBoard, this callback should be placed **before**
    the TensorBoard callback in the `model.fit` callback list
    Args:
        validation_sets (dict): A dictionary containing additional validation datasets.
            The format is `{"name": dataset_name, "data": dataset}`, where `dataset` is a TensorFlow
            dataset or generator that can be evaluated using `model.evaluate`.
        tensorboard_path (str): The file path where the TensorBoard logs should be stored.
        log_iterations (bool, optional): If True, logs evaluation metrics against training
            iterations to TensorBoard. This will track the changes of evaluation metrics over time.
            Default is True.
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
