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
    input_pca_vars_dict=False
):
    out_scale_dict = load_pickle(
        Path(setup.out_scale_dict_folder, setup.out_scale_dict_fn)
    )
    input_transform = (setup.input_sub, setup.input_div)

    if setup.ind_test_name == "pca":
        # pca_data_fn = setup.train_data_fn.split('.')[0]+f"_pca{setup.n_components}."+setup.train_data_fn.split('.')[-1]
        pca_data_fn = setup.train_data_fn.split('.')[0]+"_pca."+setup.train_data_fn.split('.')[-1]
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

    train_gen = DataGenerator(
        # data_fn=Path(setup.train_data_folder, setup.train_data_fn),
        data_fn=Path(setup.train_data_folder, train_data_fn),
        input_vars_dict=input_vars_dict,
        output_vars_dict=output_vars_dict,
        norm_fn=Path(setup.normalization_folder, setup.normalization_fn),
        input_transform=input_transform,
        output_transform=out_scale_dict,
        batch_size=setup.batch_size,
        shuffle=True,  # This feature doesn't seem to work
        do_castle=setup.do_castle_nn,
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
    input_pca_vars_dict=False
):
    out_scale_dict = load_pickle(
        Path(setup.out_scale_dict_folder, setup.out_scale_dict_fn)
    )
    input_transform = (setup.input_sub, setup.input_div)
    if test:
        data_fn = setup.test_data_folder
        filenm  = setup.test_data_fn
    else:
        data_fn = setup.train_data_folder
        filenm  = setup.valid_data_fn
    
    ngeo = nlat * nlon
    print(f"Validation batch size 'ngeo'={ngeo}.", flush=True)

    if setup.ind_test_name == "pca":
        pca_data_fn = filenm.split('.')[0]+"_pca."+filenm.split('.')[-1]
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
        batch_size=ngeo,
        shuffle=False,
        # xarray=True,
        do_castle=setup.do_castle_nn,
    )
    return valid_gen
