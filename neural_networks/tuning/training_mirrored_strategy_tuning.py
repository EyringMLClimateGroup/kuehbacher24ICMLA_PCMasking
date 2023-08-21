import os
from datetime import datetime
from pathlib import Path

import h5py
import nni
import tensorflow as tf

from neural_networks.cbrain.learning_rate_schedule import LRUpdate
from neural_networks.data_generator import build_train_generator, build_valid_generator


def train_all_models(model_descriptions, setup, tuning_params, from_checkpoint=False, continue_training=False):
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
            train_save_model(model_description, setup, tuning_params, from_checkpoint=from_checkpoint,
                             continue_training=continue_training, timestamp=timestamp)
        else:
            print(outPath + '/' + outModel, ' exists; skipping...')


def train_save_model(model_description, setup, tuning_params, from_checkpoint=False, continue_training=False,
                     timestamp=datetime.now().strftime("%Y%m%d-%H%M%S")):
    """ Train a model and save all information necessary for CAM """
    print(f"\n\nDistributed training of model {model_description}\n", flush=True)

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

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=setup.train_patience)

    report_val_loss_cb = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: nni.report_intermediate_result(logs['val_loss'])
    )
    report_val_pred_loss_cb = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: nni.report_intermediate_result(logs['val_prediction_loss'])
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=setup.train_patience)

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
        callbacks=[lrs, early_stop, report_val_loss_cb, report_val_pred_loss_cb],
        verbose=setup.train_verbose
    )
