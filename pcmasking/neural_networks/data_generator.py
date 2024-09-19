from pathlib import Path

from .cbrain.data_generator import DataGenerator
from .cbrain.utils import load_pickle


def build_train_generator(
        input_vars_dict,
        output_vars_dict,
        setup,
        num_replicas_distributed=0,
        diagnostic_mode=False,
):
    """Builds the training data generator based on input variables, output variables, and setup configuration.

    Args:
        input_vars_dict (dict): Dictionary of input variables for the model.
        output_vars_dict (dict): Dictionary of output variables for the model.
        setup (pcmasking.utils.setup.Setup): Setup configuration object containing paths, file names, and parameters.
        num_replicas_distributed (int, optional): The number of GPUs used during distributed training. Defaults to 0.
        diagnostic_mode (bool, optional): Flag to indicate if the diagnostic mode is enabled. Defaults to False.

    Returns:
        DataGenerator: A generator object for the training data.
    """
    out_scale_dict = load_pickle(
        Path(setup.out_scale_dict_folder, setup.out_scale_dict_fn)
    )
    input_transform = (setup.input_sub, setup.input_div)

    if diagnostic_mode:
        batch_size = setup.batch_size
    else:
        batch_size = compute_train_batch_size(setup, num_replicas_distributed)

    print(f"\nTraining batch size = {batch_size}.", flush=True)

    train_gen = DataGenerator(
        data_fn=Path(setup.train_data_folder, setup.train_data_fn),
        input_vars_dict=input_vars_dict,
        output_vars_dict=output_vars_dict,
        norm_fn=Path(setup.normalization_folder, setup.normalization_fn),
        input_transform=input_transform,
        output_transform=out_scale_dict,
        batch_size=batch_size,
    )
    return train_gen


def build_valid_generator(
        input_vars_dict,
        output_vars_dict,
        setup,
        nlat=64,
        nlon=128,
        test=False,
        num_replicas_distributed=0,
        diagnostic_mode=False,
):
    """Builds the validation or test data generator based on input variables, output variables, and setup configuration.

    Args:
        input_vars_dict (dict): Dictionary of input variables for the model.
        output_vars_dict (dict): Dictionary of output variables for the model.
        setup (pcmasking.utils.setup.Setup): Setup configuration object containing paths, file names, and parameters.
        nlat (int, optional): Number of latitudinal grid points. Defaults to 64.
        nlon (int, optional): Number of longitudinal grid points. Defaults to 128.
        test (bool, optional): Flag indicating whether to use test data instead of validation data. Defaults to False.
        num_replicas_distributed (int, optional): The number of GPUs used during distributed training. Defaults to 0.
        diagnostic_mode (bool, optional): Flag to indicate if the diagnostic mode is enabled. Defaults to False.

    Returns:
        DataGenerator: A generator object for the validation or test data.
    """
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
        print(f"\nTest batch size = {batch_size}.", flush=True)
    else:
        print(f"\nValidation batch size = {batch_size}.", flush=True)

    valid_gen = DataGenerator(
        data_fn=Path(data_fn, filenm),
        input_vars_dict=input_vars_dict,
        output_vars_dict=output_vars_dict,
        norm_fn=Path(setup.normalization_folder, setup.normalization_fn),
        input_transform=input_transform,
        output_transform=out_scale_dict,
        batch_size=batch_size,
    )
    return valid_gen


def build_additional_valid_generator(
        input_vars_dict,
        output_vars_dict,
        filepath,
        setup,
        nlat=64,
        nlon=128,
        num_replicas_distributed=0,
        diagnostic_mode=False,
):
    """Builds a data generator that can be used for additional validation during training.

    Args:
        input_vars_dict (dict): Dictionary of input variables for the model.
        output_vars_dict (dict): Dictionary of output variables for the model.
        filepath (str): Path to the additional data file.
        setup (pcmasking.utils.setup.Setup): Setup configuration object containing paths, file names, and parameters.
        nlat (int, optional): Number of latitudinal grid points. Defaults to 64.
        nlon (int, optional): Number of longitudinal grid points. Defaults to 128.
        num_replicas_distributed (int, optional): The number of GPUs used during distributed training. Defaults to 0.
        diagnostic_mode (bool, optional): Flag to indicate if the diagnostic mode is enabled. Defaults to False.

    Returns:
        DataGenerator: A generator object for the additional validation or test data.
    """
    out_scale_dict = load_pickle(
        Path(setup.out_scale_dict_folder, setup.out_scale_dict_fn)
    )
    input_transform = (setup.input_sub, setup.input_div)

    ngeo = nlat * nlon
    if diagnostic_mode:
        batch_size = ngeo
    else:
        batch_size = compute_val_batch_size(setup, ngeo, num_replicas_distributed)

    print(f"\nValidation batch size = {batch_size}.", flush=True)

    valid_gen = DataGenerator(
        data_fn=Path(filepath),
        input_vars_dict=input_vars_dict,
        output_vars_dict=output_vars_dict,
        norm_fn=Path(setup.normalization_folder, setup.normalization_fn),
        input_transform=input_transform,
        output_transform=out_scale_dict,
        batch_size=batch_size,
        shuffle=False,
    )
    return valid_gen


def compute_train_batch_size(setup, num_replicas_distributed):
    """Computes the training batch size based on the setup and number of GPUs.

    Args:
        setup (pcmasking.utils.setup.Setup): Setup configuration object containing batch size and strategy information.
        num_replicas_distributed (int): Number of GPUs used during distributed training.

    Returns:
        int: The global batch size for training.

    Raises:
        ValueError: If the MirroredStrategy is selected but the number of GPUs is 0.
    """
    if setup.distribute_strategy == "mirrored":
        if num_replicas_distributed == 0:
            raise ValueError("\nWARNING: Cannot run MirroredStrategy with 0 GPUs. Using 'num_replicas_distributed=1'.")

        batch_size_per_gpu = setup.batch_size
        global_batch_size = batch_size_per_gpu * num_replicas_distributed
    else:
        global_batch_size = setup.batch_size
    return global_batch_size


def compute_val_batch_size(setup, ngeo, num_replicas_distributed):
    """Computes the validation batch size based on the either a given batch size or grid size and number of GPUs.

    Args:
        setup (pcmasking.utils.setup.Setup): Setup configuration object containing validation batch size and strategy information.
        ngeo (int): Number of geographic grid points (nlat * nlon).
        num_replicas_distributed (int): Number of GPUs used during distributed training.

    Returns:
        int: The global batch size for validation.

    Raises:
        ValueError: If the MirroredStrategy is selected but the number of GPUs is 0.
    """
    if setup.use_val_batch_size:
        # Option to use validation batch size specified in setup
        # This is useful for small test runs
        return setup.val_batch_size
    else:
        if setup.distribute_strategy == "mirrored":
            if num_replicas_distributed == 0:
                raise ValueError(
                    "\nWARNING: Cannot run MirroredStrategy with 0 GPUs. Using 'num_replicas_distributed=1'.")

            global_batch_size = ngeo * num_replicas_distributed
        else:
            global_batch_size = ngeo
        return global_batch_size
