import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import nni
import tensorflow as tf

from neural_networks.cbrain.learning_rate_schedule import LRUpdate
from neural_networks.data_generator import build_train_generator, build_valid_generator
from neural_networks.cbrain.save_weights import save_norm

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

    tensorboard_log_dir = Path(model_description.get_path(setup.tensorboard_folder),
                               "{timestamp}-{filename}".format(timestamp=timestamp,
                                                               filename=model_description.get_filename()))

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        update_freq="epoch",
        profile_batch=2,
        embeddings_freq=0,
        embeddings_metadata=None,
    )

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
        train_dataset = convert_generator_to_dataset(train_gen, name="train_dataset")
        val_dataset = convert_generator_to_dataset(valid_gen, name="val_dataset")

    train_gen_input_transform = train_gen.input_transform
    train_gen_output_transform = train_gen.output_transform

    del train_gen
    del valid_gen

    history = model_description.fit_model(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=setup.epochs,
        callbacks=[lrs, early_stop, report_val_loss_cb, report_val_pred_loss_cb, tensorboard],
        verbose=setup.train_verbose,
    )

    model_description.save_model(setup.nn_output_path)

    final_metric = history.history[tuning_metric][-1]
    nni.report_final_result(final_metric)
    print(f"\nFinal {tuning_metric} is {final_metric}\n")

    # Saving norm after saving the model avoids having to create
    # the folder ourselves
    if "pca" not in model_description.model_type:
        save_norm(
            input_transform=train_gen_input_transform,
            output_transform=train_gen_output_transform,
            save_dir=save_dir,
            filename=model_description.get_filename(),
        )


def normalize(data, generator):
    data_x = data["vars"][:, generator.input_idxs]
    data_y = data["vars"][:, generator.output_idxs]

    # Normalize
    data_x = generator.input_transform.transform(data_x)
    data_y = generator.output_transform.transform(data_y)

    # Delete data to save memory
    del data

    return data_x, data_y


def convert_generator_to_dataset(generator, name):
    data = h5py.File(generator.data_fn, "r")
    batch_size = generator.batch_size

    inputs, outputs = normalize(data, generator)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs), name=name)

    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return dataset
