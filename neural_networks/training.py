import os
import pickle
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from .cbrain.learning_rate_schedule import LRUpdate
from .cbrain.save_weights import save_norm
from .data_generator import build_train_generator, build_valid_generator, build_additional_valid_generator
from .extra_validation_callback import ExtraValidation
from .load_models import load_model_weights_from_checkpoint, load_model_from_previous_training


def train_all_models(model_descriptions, setup, from_checkpoint=False, continue_training=False):
    """ Train and save all the models """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for model_description in model_descriptions:
        outModel = model_description.get_filename() + '_model.h5'
        outPath = str(model_description.get_path(setup.nn_output_path))
        if not os.path.isfile(os.path.join(outPath, outModel)):
            # todo: change for CASTLE and include from checkpoint
            train_save_model(model_description, setup, from_checkpoint=from_checkpoint,
                             continue_training=continue_training, timestamp=timestamp)
        else:
            print(outPath + '/' + outModel, ' exists; skipping...')


def train_save_model(
        model_description, setup, from_checkpoint=False, continue_training=False,
        timestamp=datetime.now().strftime("%Y%m%d-%H%M%S")
):
    """ Train a model and save all information necessary for CAM """
    print(f"\n\nTraining model {model_description}\n", flush=True)

    input_vars_dict = model_description.input_vars_dict
    output_vars_dict = model_description.output_vars_dict

    save_dir = str(model_description.get_path(setup.nn_output_path))
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nSave directory is: {str(save_dir)}\n", flush=True)

    # Load model weights from checkpoint
    if from_checkpoint:
        model_description = load_model_weights_from_checkpoint(model_description, which_checkpoint="cont")

    # Load whole model (including optimizer) from previous training
    if continue_training:
        print(f"\nContinue training for model {model_description}\n", flush=True)
        model_description.model = load_model_from_previous_training(model_description)

        previous_lr_path = Path(save_dir, "learning_rate", model_description.get_filename() + "_model_lr.p")
        print(f"\nLoading learning rate from {previous_lr_path}", flush=True)
        with open(previous_lr_path, 'rb') as f:
            init_lr = pickle.load(f)["last_lr"]
        print(f"Learning rate = {init_lr}\n", flush=True)

    else:
        init_lr = setup.init_lr

    lrs = tf.keras.callbacks.LearningRateScheduler(
        LRUpdate(init_lr=init_lr, step=setup.step_lr, divide=setup.divide_lr)
    )
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

    early_stop = EarlyStopping(monitor="val_loss", patience=setup.train_patience)

    checkpoint_dir_best = Path(save_dir, "ckpt_best", model_description.get_filename() + "_model",
                               "best_train_ckpt")
    checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir_best,
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    checkpoint_dir_cont = Path(save_dir, "ckpt_cont", model_description.get_filename() + "_model",
                               "cont_train_ckpt")
    checkpoint_cont = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir_cont,
        save_best_only=False,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
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

    # Get additional datasets
    if setup.nn_type == "castleNN" and setup.additional_val_datasets:
        additional_validation_datasets = {}
        for name_and_data in setup.additional_val_datasets:
            print(f"\nLoading additional dataset {name_and_data['name']}\n")
            data_path = name_and_data['data']
            with build_additional_valid_generator(input_vars_dict, output_vars_dict, data_path, setup) as data_gen:
                dataset = convert_generator_to_dataset(data_gen, name=name_and_data["name"] + "_dataset")
            del data_gen
            additional_validation_datasets[name_and_data["name"]] = dataset

        extra_validation = ExtraValidation(additional_validation_datasets, tensorboard_path=tensorboard_log_dir)

        callbacks = [extra_validation, lrs, tensorboard, early_stop, checkpoint_cont, checkpoint_best]
    else:
        callbacks = [lrs, tensorboard, early_stop, checkpoint_cont, checkpoint_best]

    model_description.fit_model(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=setup.epochs,
        callbacks=callbacks,
        verbose=setup.train_verbose,
    )

    model_description.save_model(setup.nn_output_path)

    # Save last learning rate
    last_lr = {"last_lr": lrs.schedule.current_lr}
    last_lr_path = Path(save_dir, "learning_rate", model_description.get_filename() + "_model_lr.p")
    Path(last_lr_path.parent).mkdir(parents=True, exist_ok=True)

    with open(last_lr_path, 'wb') as f:
        pickle.dump(last_lr, f)

    # Saving norm after saving the model avoids having to create
    # the folder ourselves
    if "pca" not in model_description.model_type:
        save_norm(
            input_transform=train_gen_input_transform,
            output_transform=train_gen_output_transform,
            save_dir=save_dir,
            filename=model_description.get_filename(),
        )

    if setup.do_sklasso_nn:
        np.savetxt(
            save_dir + f"/{model_description.get_filename()}_sklasso_coefs.txt",
            model_description.lasso_coefs,
            fmt='%1.6e',
            delimiter=",",
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

    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset
