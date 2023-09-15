import os
from datetime import datetime
from pathlib import Path

import nni
import tensorflow as tf

from neural_networks.cbrain.learning_rate_schedule import LRUpdate
from neural_networks.data_generator import build_train_generator, build_valid_generator


def train_all_models(model_descriptions, setup, tuning_params, tuning_metric='val_loss', from_checkpoint=False,
                     continue_training=False):
    """ Train and save all the models """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for model_description in model_descriptions:
        outModel = model_description.get_filename() + '_model.h5'
        outPath = str(model_description.get_path(setup.nn_output_path))
        if not os.path.isfile(os.path.join(outPath, outModel)):
            # todo: change for CASTLE and include from checkpoint
            train_save_model(model_description, setup, tuning_params, tuning_metric=tuning_metric,
                             from_checkpoint=from_checkpoint, continue_training=continue_training,
                             timestamp=timestamp)
        else:
            print(outPath + '/' + outModel, ' exists; skipping...')


def train_save_model(
        model_description, setup, tuning_params, tuning_metric, from_checkpoint=False, continue_training=False,
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
    lr_schedule_dict = tuning_params["learning_rate_schedule"]

    if lr_schedule_dict["schedule"] == "exp":
        lrs = tf.keras.callbacks.LearningRateScheduler(
            LRUpdate(init_lr=learning_rate, step=lr_schedule_dict["step"], divide=lr_schedule_dict["divide"])
        )
    elif lr_schedule_dict["schedule"] == "plateu":
        lrs = tf.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_schedule_dict["factor"],
                                             patience=3, min_lr=1e-8)
    elif lr_schedule_dict["schedule"] == "linear":
        lrs = tf.keras.callbacks.LearningRateScheduler(
            tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=lr_schedule_dict["decay_steps"],
                end_learning_rate=lr_schedule_dict["end_lr"]))

    elif lr_schedule_dict["schedule"] == "cosine":
        lrs = tf.keras.callbacks.LearningRateScheduler(
            tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=learning_rate,
                decay_steps=lr_schedule_dict["decay_steps"],
                alpha=lr_schedule_dict["alpha"]))

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=setup.train_patience)

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

        history = model_description.fit_model(
            x=train_gen,
            validation_data=valid_gen,
            epochs=setup.epochs,
            callbacks=[lrs, early_stop, report_val_loss_cb, report_val_pred_loss_cb],
            verbose=setup.train_verbose,
        )

    final_metric = history.history[tuning_metric][-1]
    nni.report_final_result(final_metric)
    print(f"\nFinal {tuning_metric} is {final_metric}\n")
