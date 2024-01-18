import os

import tensorflow as tf
from tensorflow import keras


class TemperatureDecay(keras.callbacks.Callback):
    def __init__(self, initial_temperature, decay_rate, decay_steps, warm_up=0, tb_log_dir=""):
        super().__init__()
        self.initial_temperature = initial_temperature
        print(f"\n\nInitial temperature = {self.initial_temperature}\n")

        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

        self.warm_up = warm_up

        print(f"Decay rate = {self.decay_rate}")
        print(f"Decay steps = {self.decay_steps}")
        print(f"Temperature warm up = {self.warm_up}\n\n")

        self.tb_log_dir = tb_log_dir
        self.tb_logging_active = False

        if self.tb_log_dir != "":
            self.tb_logging_active = True

            output_path = os.path.join(self.tb_log_dir, "temperature_decay")
            self.summary_writer = tf.summary.create_file_writer(output_path)

    def on_epoch_begin(self, epoch, logs=None):
        temp = self.temperature_decay(epoch)

        print(f"\nTemperature decay. Temperature = {temp}\n")
        self.model.get_layer("input_masking_layer").temp.assign(temp)

        with self.summary_writer.as_default():
            temp = self.model.get_layer("input_masking_layer").temp
            mask_sum = tf.reduce_sum(self.model.get_layer("input_masking_layer").masking_vector)

            tf.summary.scalar(name="temperature", data=temp, step=epoch)
            tf.summary.scalar(name="mask_sum", data=mask_sum, step=epoch)

    def temperature_decay(self, step):
        if step >= self.warm_up:
            return self.initial_temperature * tf.math.pow(self.decay_rate, ((step - self.warm_up) / self.decay_steps))
        else:
            return self.initial_temperature
