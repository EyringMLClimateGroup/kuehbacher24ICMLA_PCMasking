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
from pcmasking.neural_networks.callbacks.extra_validation_callback import ExtraValidation
from .load_models import load_model_weights_from_checkpoint, load_model_from_previous_training


def train_all_models(model_descriptions, setup, from_checkpoint=False, continue_training=False,
                     save_learning_rate=True):
    """
    Trains and saves all models in the given model descriptions.

    Args:
        model_descriptions (list): List of ModelDescription objects to be trained.
        setup (pcmasking.utils.setup.Setup): Setup object containing configuration details for the training.
        from_checkpoint (bool, optional): Whether to load model weights from a checkpoint. Defaults to False.
        continue_training (bool, optional): Whether to continue training from a previous training run where
            model weights were saved after training (no checkpoint). Defaults to False.
        save_learning_rate (bool, optional): Whether to save the final learning rate after training. Defaults to True.

    Returns:
        dict: A dictionary containing the training history for each model.
    """
    if setup.distribute_strategy == "mirrored":
        if any(md.strategy.num_replicas_in_sync == 0 for md in model_descriptions):
            raise EnvironmentError("Trying to run function 'train_all_models' for tf.distribute.MirroredStrategy "
                                   "but Tensorflow found no GPUs. ")

    histories = dict()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for model_description in model_descriptions:
        # This will not affect custom models which are saved as .keras files
        out_model_name = model_description.get_filename() + '_model.h5'
        out_path = str(model_description.get_path(setup.nn_output_path))

        if not os.path.isfile(os.path.join(out_path, out_model_name)):
            histories[str(model_description.output)] = train_save_model(model_description, setup,
                                                                        from_checkpoint=from_checkpoint,
                                                                        continue_training=continue_training,
                                                                        save_lr=save_learning_rate,
                                                                        timestamp=timestamp)
        else:
            print(out_path + '/' + out_model_name, ' exists; skipping...')

    return histories


def train_save_model(model_description, setup, from_checkpoint=False, continue_training=False, save_lr=True,
                     timestamp=datetime.now().strftime("%Y%m%d-%H%M%S")):
    """Trains a single model and saves necessary information for post-training usage.

    Args:
        model_description (ModelDescription): ModelDescription object for the model being trained.
        setup (pcmasking.utils.setup.Setup): Setup object containing configuration details for the training.
        from_checkpoint (bool, optional): Whether to load model weights from a checkpoint. Defaults to False.
        continue_training (bool, optional): Whether to continue training from a previous training run where
            model weights were saved after training (no checkpoint). Defaults to False
        save_lr (bool, optional): Whether to save the final learning rate after training. Defaults to True.
        timestamp (str, optional): Timestamp for naming output files. Defaults to current time.

    Returns:
        dict: A history object containing details of the training process.
    """
    if setup.distribute_strategy == "mirrored":
        num_replicas_in_sync = model_description.strategy.num_replicas_in_sync
        print(f"\n\nDistributed training of model {model_description} across {num_replicas_in_sync} GPUs\n", flush=True)
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
                               num_replicas_distributed=num_replicas_in_sync) as train_gen, \
            build_valid_generator(model_description.input_vars_dict, model_description.output_vars_dict, setup,
                                  num_replicas_distributed=num_replicas_in_sync) as valid_gen:
        train_dataset = convert_generator_to_dataset(train_gen, name="train_dataset")
        val_dataset = convert_generator_to_dataset(valid_gen, name="val_dataset")

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
    training_pcmasking = setup.nn_type in ["PreMaskNet", "MaskNet"]
    if training_pcmasking and setup.additional_val_datasets:
        additional_validation_datasets = _load_additional_datasets(model_description.input_vars_dict,
                                                                   model_description.output_vars_dict, setup,
                                                                   num_replicas_in_sync, options=options)
    else:
        additional_validation_datasets = None

    callbacks, lrs = get_callbacks(init_lr, model_description, setup, save_dir, timestamp,
                                   additional_validation_datasets)

    # Train the model
    history = model_description.fit_model(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=setup.epochs,
        callbacks=callbacks,
        verbose=setup.train_verbose,
    )

    # Save trained model
    model_description.save_model(setup.nn_output_path)

    # Save last learning rate
    if save_lr:
        _save_final_learning_rate(lrs, model_description, save_dir)

    # Saving norm after saving the model avoids having to create
    # the folder ourselves
    save_norm(
        input_transform=train_gen_input_transform,
        output_transform=train_gen_output_transform,
        save_dir=save_dir,
        filename=model_description.get_filename(),

    )

    return history


def _create_output_directory(model_description, setup):
    """Creates the output directory for saving the model and training artifacts"""
    save_dir = str(model_description.get_path(setup.nn_output_path))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nSave directory is: {str(save_dir)}\n", flush=True)
    return save_dir


def load_model_and_lr(model_description, save_dir):
    """Loads the model and its learning rate from a previous training run.

    Args:
        model_description (ModelDescription): The model for which to load the weights and learning rate.
        save_dir (str): Path to the directory where the model and learning rate are saved.

    Returns:
        float: The last learning rate from previous training.
    """
    print(f"\nContinue training for model {model_description}\n", flush=True)
    model_description.model = load_model_from_previous_training(model_description)

    previous_lr_path = Path(save_dir, "learning_rate", model_description.get_filename() + "_model_lr.p")
    print(f"\nLoading learning rate from {previous_lr_path}", flush=True)

    with open(previous_lr_path, 'rb') as f:
        init_lr = pickle.load(f)["last_lr"]

    print(f"Learning rate = {init_lr}\n", flush=True)
    return init_lr


def get_callbacks(init_lr, model_description, setup, save_dir, timestamp, additional_validation_datasets=None):
    """Creates the necessary callbacks for training the model.

    Args:
        init_lr (float): Initial learning rate for training.
        model_description (ModelDescription): The ModelDescription of the model being trained.
        setup (pcmasking.utils.setup.Setup): Setup object containing configuration details for the training.
        save_dir (str): Path to the directory where the model and logs are saved.
        timestamp (str): Timestamp for naming output files.
        additional_validation_datasets (dict, optional): Additional datasets for validation. Defaults to None.

    Returns:
        tuple: A tuple containing a list of callbacks and the learning rate scheduler.
    """
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


def _load_additional_datasets(input_vars_dict, output_vars_dict, setup, num_replicas_in_sync, options=None):
    additional_validation_datasets = {}
    for name_and_data in setup.additional_val_datasets:
        print(f"\nLoading additional dataset {name_and_data['name']}\n")
        data_path = name_and_data['data']
        with build_additional_valid_generator(input_vars_dict, output_vars_dict, data_path, setup,
                                              num_replicas_distributed=num_replicas_in_sync) as data_gen:
            dataset = convert_generator_to_dataset(data_gen, name=name_and_data["name"] + "_dataset")
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
    """Sets the learning rate schedule based on the provided configuration.

    Args:
        learning_rate (float): Initial learning rate for training.
        schedule (dict): Dictionary specifying the learning rate schedule type and parameters.

    Returns:
        object: A learning rate scheduler callback.

    Raises:
        ValueError: If an unknown schedule type is provided.
    """
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
    """Applies input and output transformations from pcmasking.neural_networks.cbrain.data_generator.DataGenerator
    to the given data.

    Args:
        data (h5py.File): HDF5 file object containing input and output variables in `data["vars"]`.
        generator (DataGenerator): DataGenerator instance to apply transformations.

    Returns:
        tuple: Transformed input and output data.
    """
    data_x = data["vars"][:, generator.input_idxs]
    data_y = data["vars"][:, generator.output_idxs]

    # Normalize
    data_x = generator.input_transform.transform(data_x)
    data_y = generator.output_transform.transform(data_y)

    # Delete data to save memory
    del data

    return data_x, data_y


def convert_generator_to_dataset(generator, name):
    """Converts a DataGenerator instance into a batched `tf.data.Dataset`.

    Args:
        generator (DataGenerator): DataGenerator instance containing the data.
        name (str): Name for the resulting dataset.

    Returns:
        tf.data.Dataset: A batched TensorFlow Dataset instance.
    """
    data = h5py.File(generator.data_fn, "r")
    batch_size = generator.batch_size

    inputs, outputs = normalize(data, generator)

    dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs), name=name)
    dataset = dataset.cache().batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return dataset
