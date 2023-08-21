import os
import pickle
from datetime import datetime
from pathlib import Path
import nni

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

from .cbrain.learning_rate_schedule import LRUpdate
from .cbrain.save_weights import save_norm
from .data_generator import build_train_generator, build_valid_generator
from .load_models import load_model_weights_from_checkpoint, load_model_from_previous_training


def train_all_models(model_descriptions, setup, tuning_params, from_checkpoint=False, continue_training=False):
    """ Train and save all the models """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for model_description in model_descriptions:
        outModel = model_description.get_filename() + '_model.h5'
        outPath = str(model_description.get_path(setup.nn_output_path))
        if not os.path.isfile(os.path.join(outPath, outModel)):
            # todo: change for CASTLE and include from checkpoint
            train_save_model(model_description, setup, tuning_params, from_checkpoint=from_checkpoint,
                             continue_training=continue_training,
                             timestamp=timestamp)
        else:
            print(outPath + '/' + outModel, ' exists; skipping...')


def train_save_model(
        model_description, setup, tuning_params, from_checkpoint=False, continue_training=False,
        timestamp=datetime.now().strftime("%Y%m%d-%H%M%S")
):
    """ Train a model and save all information necessary for CAM """
    print(f"\n\nTraining model {model_description}\n", flush=True)

    input_vars_dict = model_description.input_vars_dict
    output_vars_dict = model_description.output_vars_dict

    save_dir = str(model_description.get_path(setup.nn_output_path))
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nSave directory is: {str(save_dir)}\n", flush=True)

    learning_rate = tuning_params["learning_rate"]
    lr_schedule_tuple = tuning_params["learning_rate_schedule"]

    if lr_schedule_tuple[0] == "exp":
        step_lr = lr_schedule_tuple[1]
        divide_lr = lr_schedule_tuple[2]

        lrs = tf.keras.callbacks.LearningRateScheduler(
            LRUpdate(init_lr=learning_rate, step=step_lr, divide=divide_lr)
        )
    elif lr_schedule_tuple[0] == "plateu":
        factor = lr_schedule_tuple[1]
        lrs = tf.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=factor,
                                             patience=3, min_lr=1e-8)

    early_stop = EarlyStopping(monitor="val_loss", patience=setup.train_patience)

    report_val_loss_cb = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: nni.report_intermediate_result(logs['val_loss'])
    )
    report_val_pred_loss_cb = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: nni.report_intermediate_result(logs['val_prediction_loss'])
    )

    with build_train_generator(
            input_vars_dict, output_vars_dict, setup, input_pca_vars_dict=setup.input_pca_vars_dict,
    ) as train_gen, build_valid_generator(
        input_vars_dict, output_vars_dict, setup, input_pca_vars_dict=setup.input_pca_vars_dict,
    ) as valid_gen:

        model_description.fit_model(
            x=train_gen,
            validation_data=valid_gen,
            epochs=setup.epochs,
            callbacks=[lrs, early_stop, report_val_loss_cb, report_val_pred_loss_cb],
            verbose=setup.train_verbose,
        )
