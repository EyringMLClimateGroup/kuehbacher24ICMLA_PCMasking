import collections
import pickle
from pathlib import Path

import tensorflow as tf

from pcmasking.neural_networks.custom_models.mask_model import MaskNet
from pcmasking.neural_networks.custom_models.pre_mask_model import PreMaskNet
from pcmasking.utils.variable import Variable_Lev_Metadata


def get_path(setup, model_type, *, pc_alpha=None, threshold=None):
    """Generates a file path based on the model type and other setup parameters.

    Args:
        setup (pcmasking.utils.setup.Setup): The setup configuration object that contains paths and parameters.
        model_type (str): Type of the model, e.g., 'CausalSingleNN', 'MaskNet', etc.
        pc_alpha (float, optional): Regularization parameter for PC1. Defaults to None.
        threshold (float, optional): Causal threshold for PC1. Defaults to None.

    Returns:
        Path: A pathlib Path object representing the file path for saving or loading models.
    """
    path = Path(setup.nn_output_path, model_type)
    if model_type == "CausalSingleNN" or model_type == "CorrSingleNN":
        if setup.area_weighted:
            cfg_str = "a{pc_alpha}-t{threshold}-latwts/"
        else:
            cfg_str = "a{pc_alpha}-t{threshold}/"
        path = path / Path(
            cfg_str.format(pc_alpha=pc_alpha, threshold=threshold)
        )

    elif model_type == "MaskNet":
        cfg_str = "threshold{threshold}"

        if setup.distribute_strategy == "mirrored":
            cfg_str += "-mirrored"
        path = path / Path(cfg_str.format(threshold=setup.mask_threshold))

    elif model_type == "PreMaskNet":
        cfg_str = "lspar{lambda_sparsity}"
        if setup.distribute_strategy == "mirrored":
            cfg_str += "-mirrored"

        path = path / Path(cfg_str.format(lambda_sparsity=setup.lambda_sparsity))

    str_hl = str(setup.hidden_layers).replace(", ", "_")
    str_hl = str_hl.replace("[", "").replace("]", "")
    str_act = str(setup.activation)

    pcmasking_model = model_type in ["PreMaskNet", "MaskNet"]
    if str_act.lower() == "leakyrelu" and pcmasking_model:
        str_act += f"_{setup.relu_alpha}"

    path = path / Path(
        "hl_{hidden_layers}-act_{activation}-e_{epochs}/".format(
            hidden_layers=str_hl,
            activation=str_act,
            epochs=setup.epochs,
        )
    )
    return path


def get_filename(setup, output):
    """Generates a filename to save the model"""
    i_var = setup.output_order.index(output.var)
    i_level = output.level_idx
    if i_level is None:
        i_level = 0
    return f"{i_var}_{i_level}"


def load_model_weights_from_checkpoint(model_description, which_checkpoint):
    """Loads the model weights from the specified checkpoint.

    Args:
        model_description (pcmasking.neural_networks.models.ModelDescription): Object containing the model
            and setup parameters.
        which_checkpoint (str): Specifies which checkpoint to load, either 'best' or 'cont'.

    Raises:
        ValueError: If `which_checkpoint` is not 'best' or 'cont'.

    Returns:
        object: The model description object with loaded weights.
    """
    folder = get_checkpoint_folder(model_description.setup, model_description.model_type, model_description.output,
                                   pc_alpha=model_description.pc_alpha, threshold=model_description.threshold)
    if which_checkpoint == "best":
        ckpt_path = Path(folder, "ckpt_best",
                         get_filename(model_description.setup, model_description.output) + "_model", "best_train_ckpt")

    elif which_checkpoint == "cont":
        ckpt_path = Path(folder, "ckpt_cont",
                         get_filename(model_description.setup, model_description.output) + "_model", "cont_train_ckpt")

    else:
        raise ValueError(f"Which checkpoint value must be in ['best', 'cont']")

    print(f"\nLoading model weights from checkpoint path {ckpt_path}.\n")
    model_description.model.load_weights(ckpt_path)

    return model_description


def get_checkpoint_folder(setup, model_type, output, pc_alpha=None, threshold=None):
    folder = get_path(setup, model_type, pc_alpha=pc_alpha, threshold=threshold)

    # In case a mask threshold file was given, mask_threshold needs to be set for model loading to work
    if setup.nn_type == "MaskNet" and setup.mask_threshold is None:
        setup.mask_threshold = get_mask_net_threshold(setup, output)
        folder = get_path(setup, model_type, pc_alpha=pc_alpha, threshold=threshold)

        # Reset threshold
        setup.mask_threshold = None
    return folder


def load_model_from_previous_training(model_description):
    """Loads a previously trained model"""
    model, _ = get_model(model_description.setup, model_description.output, model_description.model_type,
                         pc_alpha=model_description.pc_alpha, threshold=model_description.threshold)
    return model


def get_model(setup, output, model_type, *, pc_alpha=None, threshold=None):
    """Retrieves the model and input list based on the setup configuration and model type.

    Args:
        setup (pcmasking.utils.setup.Setup): Setup configuration object containing paths and parameters.
        output (pcmasking.utils.variable.Variable_Lev_Metadata): Output variable metadata object.
        model_type (str): Type of model to retrieve, e.g., 'CausalSingleNN', 'MaskNet', etc.
        pc_alpha (float, optional): Regularization parameter for PC1. Defaults to None.
        threshold (float, optional): Causal threshold for PC1. Defaults to None.

    Returns:
        tuple: A tuple containing the loaded model and the input indices list.
    """
    folder = get_path(setup, model_type, pc_alpha=pc_alpha, threshold=threshold)
    filename = get_filename(setup, output)

    if setup.nn_type == "PreMaskNet":
        modelname = Path(folder, filename + '_model.keras')
        print(f"\nLoad model: {modelname}")

        model = tf.keras.models.load_model(modelname, custom_objects={'PreMaskNet': PreMaskNet})

    elif setup.nn_type == "MaskNet":
        # In case a mask threshold file was given, mask_threshold needs to be set for model loading to work
        if setup.mask_threshold is None:
            setup.mask_threshold = get_mask_net_threshold(setup, output)
            folder = get_path(setup, model_type, pc_alpha=pc_alpha, threshold=threshold)

            # Reset threshold
            setup.mask_threshold = None

        modelname = Path(folder, filename + '_model.keras')
        print(f"\nLoad model: {modelname}")

        model = tf.keras.models.load_model(modelname, custom_objects={'MaskNet': MaskNet})

    else:
        modelname = Path(folder, filename + '_model.h5')
        print(f"\nLoad model: {modelname}")

        model = tf.keras.models.load_model(modelname)

    inputs_path = Path(folder, f"{filename}_input_list.txt")
    with open(inputs_path) as inputs_file:
        input_indices = [i for i, v in enumerate(inputs_file.readlines()) if int(v)]

    return (model, input_indices)


def get_mask_net_threshold(setup, output_var):
    """Gets threshold for masking vector"""
    if setup.mask_threshold is not None:
        # Single float threshold given
        threshold = setup.mask_threshold
    elif setup.mask_threshold_file is not None:
        # Dictionary with threshold values per output variable given
        print(f"\nLoading threshold file {(Path(*Path(setup.mask_threshold_file).parts[-4:]))}\n")

        with open(setup.mask_threshold_file, "rb") as in_file:
            mask_threshold_file = pickle.load(in_file)
            threshold = mask_threshold_file[output_var]
    return threshold


def get_var_list(setup, target_vars):
    """Generates a list of output variables"""
    output_list = list()
    for spcam_var in target_vars:
        if spcam_var.dimensions == 3:
            var_levels = [setup.children_idx_levs, setup.parents_idx_levs] \
                [spcam_var.type == 'in']
            for level, _ in var_levels:
                var_name = f"{spcam_var.name}-{round(level, 2)}"
                output_list.append(var_name)
        elif spcam_var.dimensions == 2:
            var_name = spcam_var.name
            output_list.append(var_name)
    return output_list


def load_single_model(setup, var_name):
    """Loads a single model based on the setup configuration and variable name.

    Args:
        setup (pcmasking.utils.setup.Setup): Setup configuration object containing model parameters.
        var_name (str): Name of the variable to load the model for.

    Returns:
        dict: A dictionary containing the loaded model.

    Raises:
        NotImplementedError: If the neural network type in setup is not supported.
    """
    loading_pcmasking = setup.nn_type in ["PreMaskNet", "MaskNet"]

    if setup.do_single_nn or setup.do_random_single_nn or loading_pcmasking:
        var = Variable_Lev_Metadata.parse_var_name(var_name)
        return {var: get_model(setup, var, setup.nn_type, pc_alpha=None, threshold=None)}

    if setup.do_causal_single_nn:
        models = {}
        var = Variable_Lev_Metadata.parse_var_name(var_name)

        for pc_alpha in setup.pc_alphas:
            nn_type = 'CausalSingleNN' if setup.ind_test_name == 'parcorr' else 'CorrSingleNN'
            models[pc_alpha] = {}

            for threshold in setup.thresholds:
                models[pc_alpha][threshold] = {}
                models[pc_alpha][threshold][var] = get_model(setup, var, nn_type, pc_alpha=pc_alpha,
                                                             threshold=threshold)
        return models
    else:
        raise NotImplementedError(f"load_single_model is not implemented for neural network type {setup.nn_type}")


def load_models(setup, skip_causal_phq=False):
    """Loads all neural network models specified in the setup configuration.

    Args:
        setup (pcmasking.utils.setup.Setup): Setup configuration object containing model parameters.
        skip_causal_phq (bool, optional): Whether to skip the upper two levels of moistening tendencies.
            When using a causal discovery pre-step, these models usually don't exist because no
            causal parents are found for these ouputs. Defaults to False.

    Returns:
        dict: A dictionary containing all loaded models.
    """
    models = collections.defaultdict(dict)

    output_list = get_var_list(setup, setup.spcam_outputs)
    pcmasking_model = setup.nn_type in ["PreMaskNet", "MaskNet"]

    if setup.do_single_nn or setup.do_random_single_nn or pcmasking_model:
        nn_type = setup.nn_type
        for output in output_list:
            output = Variable_Lev_Metadata.parse_var_name(output)
            models[nn_type][output] = get_model(
                setup,
                output,
                nn_type,
                pc_alpha=None,
                threshold=None
            )
    if setup.do_causal_single_nn:
        for pc_alpha in setup.pc_alphas:
            nn_type = 'CausalSingleNN' if setup.ind_test_name == 'parcorr' else 'CorrSingleNN'
            models[nn_type][pc_alpha] = {}
            for threshold in setup.thresholds:
                models[nn_type][pc_alpha][threshold] = {}
                for output in output_list:
                    output = Variable_Lev_Metadata.parse_var_name(output)
                    # This skips the first two networks for moistening tendencies when
                    # loading CausalSingleNN, because there were no causal parents found for these outputs
                    if skip_causal_phq and (str(output) == "phq-3.64" or str(output) == "phq-7.59"):
                        continue
                    models[nn_type][pc_alpha][threshold][output] = get_model(
                        setup,
                        output,
                        nn_type,
                        pc_alpha=pc_alpha,
                        threshold=threshold
                    )

    return models


def get_save_plot_folder(setup, model_type, output, *, pc_alpha=None, threshold=None):
    """ Generates a folder path to save diagnostic plots for the model"""
    folder = get_path(setup, model_type, pc_alpha=pc_alpha, threshold=threshold)
    path = Path(folder, 'diagnostics')
    return path
