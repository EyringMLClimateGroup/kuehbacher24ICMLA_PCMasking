from pathlib import Path
import os.path

from .cbrain.data_generator import DataGenerator
from .cbrain.utils import load_pickle
from .pca import pca


def build_train_generator(
        input_vars_dict,
        output_vars_dict,
        setup,
        # save_dir=False,
        input_pca_vars_dict=False,
        num_replicas_distributed=0,  # the number of GPUs when training was done in parallel
        diagnostic_mode=False,
):
    out_scale_dict = load_pickle(
        Path(setup.out_scale_dict_folder, setup.out_scale_dict_fn)
    )
    input_transform = (setup.input_sub, setup.input_div)

    if setup.ind_test_name == "pca":
        # pca_data_fn = setup.train_data_fn.split('.')[0]+f"_pca{setup.n_components}."+setup.train_data_fn.split('.')[-1]
        pca_data_fn = setup.train_data_fn.split('.')[0] + "_pca." + setup.train_data_fn.split('.')[-1]
        if not os.path.exists(Path(setup.train_data_folder, pca_data_fn)):
            print(f"Creating training PC-components...")
            pca(
                data_fn=Path(setup.train_data_folder, setup.train_data_fn),
                pca_data_fn=Path(setup.train_data_folder, pca_data_fn),
                input_vars_dict=input_vars_dict,
                norm_fn=Path(setup.normalization_folder, setup.normalization_fn),
                load_pca_model=False,
                # save_dir=save_dir,
                setup=setup,
            )
        # setup.train_data_fn = pca_data_fn
        train_data_fn = pca_data_fn
        input_transform = None
        input_vars_dict = input_pca_vars_dict
    else:
        train_data_fn = setup.train_data_fn

    if diagnostic_mode:
        batch_size = setup.batch_size
    else:
        batch_size = compute_train_batch_size(setup, num_replicas_distributed)

    print(f"Training batch size = {batch_size}.", flush=True)

    train_gen = DataGenerator(
        # data_fn=Path(setup.train_data_folder, setup.train_data_fn),
        data_fn=Path(setup.train_data_folder, train_data_fn),
        input_vars_dict=input_vars_dict,
        output_vars_dict=output_vars_dict,
        norm_fn=Path(setup.normalization_folder, setup.normalization_fn),
        input_transform=input_transform,
        output_transform=out_scale_dict,
        batch_size=batch_size,
        shuffle=True,  # This feature doesn't seem to work
        input_y=True if setup.nn_type == "CASTLEOriginal" else False,
    )
    return train_gen


def build_valid_generator(
        input_vars_dict,
        output_vars_dict,
        setup,
        nlat=64,
        nlon=128,
        test=False,
        # save_dir=False,
        input_pca_vars_dict=False,
        num_replicas_distributed=0,  # the number of GPUs when training was done in parallel
        diagnostic_mode=False,
):
    out_scale_dict = load_pickle(
        Path(setup.out_scale_dict_folder, setup.out_scale_dict_fn)
    )
    input_transform = (setup.input_sub, setup.input_div)
    if test:
        data_fn = setup.test_data_folder
        filenm = setup.test_data_fn
    else:
        data_fn = setup.train_data_folder
        filenm = setup.valid_data_fn

    ngeo = nlat * nlon
    if diagnostic_mode:
        batch_size = ngeo
    else:
        batch_size = compute_val_batch_size(setup, ngeo, num_replicas_distributed)

    if test:
        print(f"Test batch size = {batch_size}.", flush=True)
    else:
        print(f"Validation batch size = {batch_size}.", flush=True)

    if setup.ind_test_name == "pca":
        pca_data_fn = filenm.split('.')[0] + "_pca." + filenm.split('.')[-1]
        if not os.path.exists(Path(setup.train_data_folder, pca_data_fn)):
            print(f"Creating validating/test PC-components...")
            pca(
                data_fn=Path(data_fn, filenm),
                pca_data_fn=Path(data_fn, pca_data_fn),
                input_vars_dict=input_vars_dict,
                norm_fn=Path(setup.normalization_folder, setup.normalization_fn),
                load_pca_model=True,
                # save_dir=save_dir,
                setup=setup,
            )
        filenm = pca_data_fn
        input_transform = None
        input_vars_dict = input_pca_vars_dict

    valid_gen = DataGenerator(
        data_fn=Path(data_fn, filenm),
        input_vars_dict=input_vars_dict,
        output_vars_dict=output_vars_dict,
        norm_fn=Path(setup.normalization_folder, setup.normalization_fn),
        input_transform=input_transform,
        output_transform=out_scale_dict,
        batch_size=batch_size,
        shuffle=False,
        input_y=True if setup.nn_type == "CASTLEOriginal" else False,
        # xarray=True,
    )
    return valid_gen


def build_additional_valid_generator(
        input_vars_dict,
        output_vars_dict,
        filepath,
        setup,
        nlat=64,
        nlon=128,
        test=False,
        # save_dir=False,
        num_replicas_distributed=0,  # the number of GPUs when training was done in parallel
        diagnostic_mode=False,
):
    out_scale_dict = load_pickle(
        Path(setup.out_scale_dict_folder, setup.out_scale_dict_fn)
    )
    input_transform = (setup.input_sub, setup.input_div)

    ngeo = nlat * nlon
    if diagnostic_mode:
        batch_size = ngeo
    else:
        batch_size = compute_val_batch_size(setup, ngeo, num_replicas_distributed)

    if test:
        print(f"Test batch size = {batch_size}.", flush=True)
    else:
        print(f"Validation batch size = {batch_size}.", flush=True)

    valid_gen = DataGenerator(
        data_fn=Path(filepath),
        input_vars_dict=input_vars_dict,
        output_vars_dict=output_vars_dict,
        norm_fn=Path(setup.normalization_folder, setup.normalization_fn),
        input_transform=input_transform,
        output_transform=out_scale_dict,
        batch_size=batch_size,
        shuffle=False,
        input_y=True if setup.nn_type == "CASTLEOriginal" else False,
    # xarray=True,
    )
    return valid_gen


def compute_train_batch_size(setup, num_replicas_distributed):
    if setup.distribute_strategy == "mirrored":
        if num_replicas_distributed == 0:
            raise ValueError("\nWARNING: Cannot run MirroredStrategy with 0 GPUs. Using 'num_replicas_distributed=1'.")

        batch_size_per_gpu = setup.batch_size
        global_batch_size = batch_size_per_gpu * num_replicas_distributed
    else:
        global_batch_size = setup.batch_size
    return global_batch_size


def compute_val_batch_size(setup, ngeo, num_replicas_distributed):
    if setup.use_val_batch_size:
        # Option to use validation batch size specified in setup
        # This is useful for small test runs
        return setup.val_batch_size
    else:
        if setup.distribute_strategy == "mirrored":
            if num_replicas_distributed == 0:
                raise ValueError("\nWARNING: Cannot run MirroredStrategy with 0 GPUs. Using 'num_replicas_distributed=1'.")

            global_batch_size = ngeo * num_replicas_distributed
        else:
            global_batch_size = ngeo
        return global_batch_size
