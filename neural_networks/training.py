import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

from .cbrain.learning_rate_schedule import LRUpdate
from .cbrain.save_weights import save_norm
from .data_generator import build_train_generator, build_valid_generator
from .load_models import load_model_weights_from_checkpoint


def train_all_models(model_descriptions, setup, from_checkpoint=False):
    """ Train and save all the models """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for model_description in model_descriptions:
        outModel = model_description.get_filename() + '_model.h5'
        outPath = str(model_description.get_path(setup.nn_output_path))
        if not os.path.isfile(os.path.join(outPath, outModel)):
            # todo: change for CASTLE and include from checkpoint
            train_save_model(model_description, setup, from_checkpoint, timestamp)
        else:
            print(outPath + '/' + outModel, ' exists; skipping...')


def train_save_model(
        model_description, setup,  from_checkpoint=False, timestamp=datetime.now().strftime("%Y%m%d-%H%M%S")
):
    """ Train a model and save all information necessary for CAM """
    print(f"\n\nTraining model {model_description}\n", flush=True)

    input_vars_dict = model_description.input_vars_dict
    output_vars_dict = model_description.output_vars_dict

    save_dir = str(model_description.get_path(setup.nn_output_path))
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nSave directory is: {str(save_dir)}\n", flush=True)

    # If this is the continuation of a previous training, load the model weights
    if from_checkpoint:
        print(f"\nLoading model weights from checkpoint.\n")
        load_model_weights_from_checkpoint(model_description, which_checkpoint="cont")

    with build_train_generator(
            input_vars_dict, output_vars_dict, setup, input_pca_vars_dict=setup.input_pca_vars_dict,
    ) as train_gen, build_valid_generator(
        input_vars_dict, output_vars_dict, setup, input_pca_vars_dict=setup.input_pca_vars_dict,
    ) as valid_gen:

        lrs = LearningRateScheduler(
            LRUpdate(init_lr=setup.init_lr, step=setup.step_lr, divide=setup.divide_lr)
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
            save_best_only=True,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            verbose=1
        )

        model_description.fit_model(
            x=train_gen,
            validation_data=valid_gen,
            epochs=setup.epochs,
            #             callbacks=[lrs, tensorboard, early_stop],
            callbacks=[lrs, tensorboard, early_stop, checkpoint_cont, checkpoint_best],
            verbose=setup.train_verbose,
        )

        model_description.save_model(setup.nn_output_path)
        # Saving norm after saving the model avoids having to create
        # the folder ourselves
        if "pca" not in model_description.model_type:
            save_norm(
                input_transform=train_gen.input_transform,
                output_transform=train_gen.output_transform,
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
