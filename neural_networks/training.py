import os
import pickle
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

from .cbrain.learning_rate_schedule import LRUpdate
from .cbrain.save_weights import save_norm
from .data_generator import build_train_generator, build_valid_generator, build_additional_valid_generator
from .extra_validation_callback import ExtraValidation
from .load_models import load_model_weights_from_checkpoint, load_model_from_previous_training


def train_all_models(model_descriptions, setup, from_checkpoint=False, continue_training=False):
    """ Train and save all the models """
    if setup.distribute_strategy == "mirrored":
        if any(md.strategy.num_replicas_in_sync == 0 for md in model_descriptions):
            raise EnvironmentError("Trying to run function 'train_all_models' for tf.distribute.MirroredStrategy "
                                   "but Tensorflow found no GPUs. ")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for model_description in model_descriptions:
        # todo: include CASTLE models?
        # This will not affect CASTLE models which are saved as .keras files
        out_model_name = model_description.get_filename() + '_model.h5'
        out_path = str(model_description.get_path(setup.nn_output_path))
        if not os.path.isfile(os.path.join(out_path, out_model_name)):
            train_save_model(model_description, setup, from_checkpoint=from_checkpoint,
                             continue_training=continue_training, timestamp=timestamp)
        else:
            print(out_path + '/' + out_model_name, ' exists; skipping...')


def train_save_model(
        model_description, setup, from_checkpoint=False, continue_training=False,
        timestamp=datetime.now().strftime("%Y%m%d-%H%M%S")):
    """ Train a model and save all information necessary for CAM """
    if setup.distribute_strategy == "mirrored":
        print(f"\n\nDistributed training of model {model_description}\n", flush=True)
        num_replicas_in_sync = model_description.strategy.num_replicas_in_sync
    else:
        print(f"\n\nTraining model {model_description}\n", flush=True)
        num_replicas_in_sync = 0

    save_dir = _create_output_directory(model_description, setup)

    # Load model weights from checkpoint
    if from_checkpoint:
        model_description = load_model_weights_from_checkpoint(model_description, which_checkpoint="cont")

    # Load whole model (including optimizer) from previous training
    if continue_training:
        init_lr = load_model_and_lr(model_description, save_dir)
    else:
        init_lr = setup.init_lr

    # Convert DataGenerator to dataset (more flexible)
    with build_train_generator(model_description.input_vars_dict, model_description.output_vars_dict, setup,
                               input_pca_vars_dict=setup.input_pca_vars_dict,
                               num_replicas_distributed=num_replicas_in_sync) as train_gen, \
            build_valid_generator(model_description.input_vars_dict, model_description.output_vars_dict, setup,
                                  input_pca_vars_dict=setup.input_pca_vars_dict,
                                  num_replicas_distributed=num_replicas_in_sync) as valid_gen:
        train_dataset = convert_generator_to_dataset(train_gen, name="train_dataset",
                                                     input_y=True if setup.nn_type == "CASTLEOriginal" else False)
        val_dataset = convert_generator_to_dataset(valid_gen, name="val_dataset",
                                                   input_y=True if setup.nn_type == "CASTLEOriginal" else False)

        train_gen_input_transform = train_gen.input_transform
        train_gen_output_transform = train_gen.output_transform

        del train_gen
        del valid_gen

    # Set distribute strategy for parallelized training
    options = None
    if setup.distribute_strategy == "mirrored":
        # Batch and set sharding policy
        # Default is AUTO, but we want DATA
        # See https://www.tensorflow.org/api_docs/python/tf/data/experimental/AutoShardPolicy
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        train_dataset = train_dataset.with_options(options)
        val_dataset = val_dataset.with_options(options)

    # Setup callbacks
    if (setup.nn_type == "CASTLEOriginal" or setup.nn_type == "CASTLEAdapted") and setup.additional_val_datasets:
        additional_validation_datasets = _load_additional_datasets(model_description.input_vars_dict,
                                                                   model_description.output_vars_dict, setup,
                                                                   options=options)

    callbacks, lrs = get_callbacks(init_lr, model_description, setup, save_dir, timestamp,
                                   additional_validation_datasets)

    # Train the model
    model_description.fit_model(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=setup.epochs,
        callbacks=callbacks,
        verbose=setup.train_verbose,
    )

    # Save trained model
    model_description.save_model(setup.nn_output_path)

    # Save last learning rate
    _save_final_learning_rate(lrs, model_description, save_dir)

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


def _create_output_directory(model_description, setup):
    save_dir = str(model_description.get_path(setup.nn_output_path))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nSave directory is: {str(save_dir)}\n", flush=True)
    return save_dir


def load_model_and_lr(model_description, save_dir):
    print(f"\nContinue training for model {model_description}\n", flush=True)
    model_description.model = load_model_from_previous_training(model_description)

    previous_lr_path = Path(save_dir, "learning_rate", model_description.get_filename() + "_model_lr.p")
    print(f"\nLoading learning rate from {previous_lr_path}", flush=True)

    with open(previous_lr_path, 'rb') as f:
        init_lr = pickle.load(f)["last_lr"]

    print(f"Learning rate = {init_lr}\n", flush=True)
    return init_lr


def get_callbacks(init_lr, model_description, setup, save_dir, timestamp, additional_validation_datasets=None):
    lrs = set_learning_rate_schedule(init_lr, setup.lr_schedule)

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

    if additional_validation_datasets is not None:
        extra_validation_cb = ExtraValidation(additional_validation_datasets, tensorboard_path=tensorboard_log_dir)

        callbacks = [extra_validation_cb, lrs, tensorboard, early_stop, checkpoint_cont, checkpoint_best]
    else:
        callbacks = [lrs, tensorboard, early_stop, checkpoint_cont, checkpoint_best]
    return callbacks, lrs


def _load_additional_datasets(input_vars_dict, output_vars_dict, setup, options=None):
    additional_validation_datasets = {}
    for name_and_data in setup.additional_val_datasets:
        print(f"\nLoading additional dataset {name_and_data['name']}\n")
        data_path = name_and_data['data']
        with build_additional_valid_generator(input_vars_dict, output_vars_dict, data_path, setup) as data_gen:
            dataset = convert_generator_to_dataset(data_gen, name=name_and_data["name"] + "_dataset", \
                                                   input_y=True if setup.nn_type == "CASTLEOriginal" else False, )
        del data_gen

        if options is not None:
            dataset = dataset.with_options(options)
        additional_validation_datasets[name_and_data["name"]] = dataset

    return additional_validation_datasets


def _save_final_learning_rate(lrs, model_description, save_dir):
    last_lr = {"last_lr": lrs.schedule.current_lr}
    last_lr_path = Path(save_dir, "learning_rate", model_description.get_filename() + "_model_lr.p")
    Path(last_lr_path.parent).mkdir(parents=True, exist_ok=True)
    with open(last_lr_path, 'wb') as f:
        pickle.dump(last_lr, f)


def set_learning_rate_schedule(learning_rate, schedule):
    if schedule["schedule"] == "exponential":
        lrs = tf.keras.callbacks.LearningRateScheduler(
            LRUpdate(init_lr=learning_rate, step=schedule["step"], divide=schedule["divide"])
        )
    elif schedule["schedule"] == "plateau":
        lrs = tf.callbacks.ReduceLROnPlateau(monitor=schedule["monitor"], factor=schedule["factor"],
                                             patience=schedule["patience"], min_lr=schedule["min_lr"])
    elif schedule["schedule"] == "linear":
        lrs = tf.keras.callbacks.LearningRateScheduler(
            tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=schedule["decay_steps"],
                end_learning_rate=schedule["end_lr"]))

    elif schedule["schedule"] == "cosine":
        lrs = tf.keras.callbacks.LearningRateScheduler(
            tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=learning_rate,
                decay_steps=schedule["decay_steps"],
                alpha=schedule["alpha"]))
    else:
        raise ValueError(f"Unknown value for learning rate schedule: {schedule['schedule']}.")
    return lrs


def normalize(data, generator):
    """Applies input and output transformations from
    neural_networks.cbrain.data_generator.DataGenerator instance to
    input and output variables in `data`.

    Args:
        data: h5py.File object containing input and output variables in data["vars"]
        generator (neural_networks.cbrain.data_generator.DataGenerator): DataGenerator instance
    """
    data_x = data["vars"][:, generator.input_idxs]
    data_y = data["vars"][:, generator.output_idxs]

    # Normalize
    data_x = generator.input_transform.transform(data_x)
    data_y = generator.output_transform.transform(data_y)

    # Delete data to save memory
    del data

    return data_x, data_y


def convert_generator_to_dataset(generator, name, input_y=False):
    """Converts an instance of neural_networks.cbrain.data_generator.DataGenerator into
    a batched tf.data.Dataset object.

    Args:
        generator (neural_networks.cbrain.data_generator.DataGenerator): DataGenerator instance
        name: Name of the output dataset
        input_y: Whether the target y is part of the network inputs.

    Returns:
        Batched tf.data.Dataset instance
    """
    data = h5py.File(generator.data_fn, "r")
    batch_size = generator.batch_size

    inputs, outputs = normalize(data, generator)

    if input_y:
        outputs_inputs = np.concatenate((outputs, inputs), axis=1)
        dataset = tf.data.Dataset.from_tensor_slices(outputs_inputs, name=name)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs), name=name)

    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return dataset
