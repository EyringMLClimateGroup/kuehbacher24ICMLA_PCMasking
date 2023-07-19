import os
import h5py
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint

from .cbrain.learning_rate_schedule import LRUpdate
from .cbrain.save_weights import save_norm
from .data_generator import build_train_generator, build_valid_generator


def train_all_models(model_descriptions, setup):
    """ Train and save all the models """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for model_description in model_descriptions:
        outModel = model_description.get_filename() + '_model.h5'
        outPath = str(model_description.get_path(setup.nn_output_path))
        if not os.path.isfile(os.path.join(outPath, outModel)):
            if setup.do_mirrored_strategy and model_description.strategy.num_replicas_in_sync == 0:
                print(f"Model {model_description} cannot be trained with tf.distribute.MirroredStrategy "
                      f"because Tensorflow found no GPUs. Skipping ... ")
            else:
                train_save_model(model_description, setup, timestamp)
        else:
            print(outPath + '/' + outModel, ' exists; skipping...')


def train_save_model(model_description, setup, timestamp=datetime.now().strftime("%Y%m%d-%H%M%S")):
    """ Train a model and save all information necessary for CAM """
    print(f"\n\nDistributed training of model {model_description}\n", flush=True)

    input_vars_dict = model_description.input_vars_dict
    output_vars_dict = model_description.output_vars_dict

    save_dir = str(model_description.get_path(setup.nn_output_path))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    def normalize(data, generator):
        data_x = data["vars"][:, generator.input_idxs]
        data_y = data["vars"][:, generator.output_idxs]

        # Normalize
        data_x = generator.input_transform.transform(data_x)
        data_y = generator.output_transform.transform(data_y)

        if setup.do_castle_nn:
            return data_x, np.concatenate([data_y, data_x], axis=1)

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

    # Adjust learning rate for distributed training
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

    checkpoint = ModelCheckpoint(
        str(model_description.get_path(setup.nn_output_path)),
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    model_description.fit_model(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=setup.epochs,
        #             callbacks=[lrs, tensorboard, early_stop],
        callbacks=[lrs, tensorboard, early_stop, checkpoint],
        verbose=setup.train_verbose
    )

    model_description.save_model(setup.nn_output_path)

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
