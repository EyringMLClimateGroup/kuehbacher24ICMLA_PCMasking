import os
import pickle
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

from .cbrain.learning_rate_schedule import LRUpdate
from .cbrain.save_weights import save_norm
from .data_generator import build_train_generator, build_valid_generator
from .load_models import load_model_weights_from_checkpoint, load_model_from_previous_training


def train_all_models(model_descriptions, setup, from_checkpoint=False, continue_training=False):
    """ Train and save all the models """
    if setup.distribute_strategy == "mirrored":
        if any(md.strategy.num_replicas_in_sync == 0 for md in model_descriptions):
            raise ValueError("Trying to run function 'train_all_models' for tf.distribute.MirroredStrategy "
                             "but Tensorflow found no GPUs. ")
    elif setup.distribute_strategy == "multi_worker_mirrored":
        n_workers = int(os.environ['SLURM_NTASKS'])
        if n_workers == 0:
            raise ValueError("Trying to run function 'train_all_models' for tf.distribute.MultiWorkerMirroredStrategy "
                             "but no SLURM tasks were found. ")
    else:
        raise ValueError("Trying to run function 'train_all_models' with distributed training "
                         "but no distribute strategy is not set in configuration file. ")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for model_description in model_descriptions:
        # todo: change for CASTLE and include from checkpoint
        outModel = model_description.get_filename() + '_model.h5'
        outPath = str(model_description.get_path(setup.nn_output_path))
        if not os.path.isfile(os.path.join(outPath, outModel)):
            train_save_model(model_description, setup, from_checkpoint=from_checkpoint,
                             continue_training=continue_training, timestamp=timestamp)
        else:
            print(outPath + '/' + outModel, ' exists; skipping...')


def train_save_model(model_description, setup, from_checkpoint=False, continue_training=False,
                     timestamp=datetime.now().strftime("%Y%m%d-%H%M%S")):
    """ Train a model and save all information necessary for CAM """
    print(f"\n\nDistributed training of model {model_description}\n", flush=True)

    input_vars_dict = model_description.input_vars_dict
    output_vars_dict = model_description.output_vars_dict

    save_dir = str(model_description.get_path(setup.nn_output_path))
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nSave directory is: {str(save_dir)}\n", flush=True)

    # If this is the continuation of a previous training, load the model weights
    if from_checkpoint:
        print(f"\nLoading model weights from checkpoint.\n")
        load_model_weights_from_checkpoint(model_description, which_checkpoint="cont")

    if continue_training:
        print(f"\nContinue training for model {model_description}\n", flush=True)
        model_description.model = load_model_from_previous_training(model_description)

        previous_lr_path = Path(save_dir, "learning_rate", model_description.get_filename() + "_model_lr.p")
        print(f"\nLoading learning rate from {previous_lr_path}", flush=True)
        with open(previous_lr_path, 'rb') as f:
            previous_lr = pickle.load(f)["last_lr"]
        print(f"Learning rate = {previous_lr}\n", flush=True)

        # Adjust learning rate for distributed training
        init_lr = previous_lr * model_description.strategy.num_replicas_in_sync
    else:
        init_lr = setup.init_lr * model_description.strategy.num_replicas_in_sync

    lrs = LearningRateScheduler(
        LRUpdate(init_lr=init_lr, step=setup.step_lr, divide=setup.divide_lr)
    )

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=Path(
            model_description.get_path(setup.tensorboard_folder),
            "{timestamp}-{filename}".format(
                timestamp=timestamp, filename=model_description.get_filename()
            ),
        ),
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        update_freq="epoch",
        profile_batch=2,
        embeddings_freq=0,
        embeddings_metadata=None,
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=setup.train_patience)

    checkpoint_dir_best = Path(save_dir, "ckpt_best", model_description.get_filename() + "_model", "best_train_ckpt")
    checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir_best,
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    checkpoint_dir_cont = Path(save_dir, "ckpt_cont", model_description.get_filename() + "_model", "cont_train_ckpt")
    checkpoint_cont = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir_cont,
        save_best_only=False,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
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

    with build_train_generator(input_vars_dict, output_vars_dict, setup, input_pca_vars_dict=setup.input_pca_vars_dict,
                               num_replicas_distributed=model_description.strategy.num_replicas_in_sync) as train_gen, \
            build_valid_generator(input_vars_dict, output_vars_dict, setup,
                                  input_pca_vars_dict=setup.input_pca_vars_dict,
                                  num_replicas_distributed=model_description.strategy.num_replicas_in_sync) as valid_gen:
        train_data = h5py.File(train_gen.data_fn, "r")
        val_data = h5py.File(valid_gen.data_fn, "r")

        train_batch_size = train_gen.batch_size
        val_batch_size = valid_gen.batch_size

        train_data_inputs, train_data_outputs = normalize(train_data, train_gen)
        val_data_inputs, val_data_outputs = normalize(val_data, valid_gen)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_data_inputs, train_data_outputs),
                                                           name="train_dataset")
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data_inputs, val_data_outputs), name="val_dataset")

    train_gen_input_transform = train_gen.input_transform
    train_gen_output_transform = train_gen.output_transform

    del train_gen
    del valid_gen

    # Batch and set sharding policy
    # Default is AUTO, but we want DATA
    # See https://www.tensorflow.org/api_docs/python/tf/data/experimental/AutoShardPolicy
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    train_dataset = train_dataset.batch(train_batch_size, drop_remainder=True).with_options(options)
    val_dataset = val_dataset.batch(val_batch_size, drop_remainder=True).with_options(options)

    model_description.fit_model(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=setup.epochs,
        #             callbacks=[lrs, tensorboard, early_stop],
        callbacks=[lrs, tensorboard, early_stop, checkpoint_best, checkpoint_cont],
        verbose=setup.train_verbose
    )

    if setup.distribute_strategy == "multi_worker_mirrored":
        task_type, task_id = (model_description.strategy.cluster_resolver.task_type,
                              model_description.strategy.cluster_resolver.task_id)
        print(f"\nMultiworker task_type={task_type}, task_id={task_type}")

        # Apparently, it is important to save model in all workers, not just the chief
        write_model_path = write_filepath(setup.nn_output_path, task_type, task_id)
        model_description.save_model(write_model_path)
        # Delete temporary models from the works
        if not _is_chief(task_type, task_id):
            tf.io.gfile.rmtree(os.path.dirname(write_model_path))
    else:
        model_description.save_model(setup.nn_output_path)

    # Save last learning rate
    last_lr = {"last_lr": lrs.schedule.current_lr}
    last_lr_path = Path(save_dir, "learning_rate", model_description.get_filename() + "_model_lr.p")
    Path(last_lr_path.parent).mkdir(parents=True, exist_ok=True)

    with open(last_lr_path, 'wb') as f:
        pickle.dump(last_lr, f)

    # Saving norm after saving the model avoids having to create the folder ourselves
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


def _is_chief(task_type, task_id):
    # Note: there are two possible `TF_CONFIG` configurations.
    #   1) In addition to `worker` tasks, a `chief` task type is use;
    #      in this case, this function should be modified to
    #      `return task_type == 'chief'`.
    #   2) Only `worker` task type is used; in this case, worker 0 is
    #      regarded as the chief. The implementation demonstrated here
    #      is for this case.
    # For the purpose of this Colab section, the `task_type` is `None` case
    # is added because it is effectively run with only a single worker.
    return (task_type == 'worker' and task_id == 0) or task_type is None


def _get_temp_dir(dirpath, task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir


def write_filepath(filepath, task_type, task_id):
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not _is_chief(task_type, task_id):
        dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)
