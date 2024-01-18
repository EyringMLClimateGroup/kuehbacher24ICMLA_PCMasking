import collections
from pathlib import Path

import tensorflow as tf

from utils.variable import Variable_Lev_Metadata
from neural_networks.castle.castle_model_adapted import CASTLEAdapted
from neural_networks.castle.castle_model_original import CASTLEOriginal
from neural_networks.castle.castle_model_simplified import CASTLESimplified
from neural_networks.castle.gumbel_softmax_single_output_model import GumbelSoftmaxSingleOutputModel
from neural_networks.castle.legacy.castle_model import CASTLE
from neural_networks.castle.layers.masked_dense_layer import MaskedDenseLayer
from neural_networks.castle.layers.gumbel_softmax_layer import StraightThroughGumbelSoftmaxMaskingLayer


def get_path(setup, model_type, *, pc_alpha=None, threshold=None):
    """ Generate a path based on this model metadata """
    path = Path(setup.nn_output_path, model_type)
    if model_type == "CausalSingleNN" or model_type == "CorrSingleNN":
        if setup.area_weighted:
            cfg_str = "a{pc_alpha}-t{threshold}-latwts/"
        else:
            cfg_str = "a{pc_alpha}-t{threshold}/"
        path = path / Path(
            cfg_str.format(pc_alpha=pc_alpha, threshold=threshold)
        )
    # elif model_type == "pcaNN":
    #     if setup.area_weighted:
    #         cfg_str = "pcs{n_components}-latwts/" 
    #     else: 
    #         cfg_str = "pcs{n_components}/"
    #     path = path / Path(
    #         cfg_str.format(n_components=setup.n_components)
    #     )
    elif model_type == "pcaNN":
        cfg_str = "pcs{n_components}/"
        path = path / Path(
            cfg_str.format(n_components=setup.n_components)
        )
    elif "lasso" in model_type:
        cfg_str = "a{alpha_lasso}/"
        path = path / Path(
            cfg_str.format(alpha_lasso=setup.alpha_lasso)
        )
    elif model_type == "CASTLEOriginal":
        cfg_str = "r{rho}-a{alpha}-b{beta}-l{lambda_weight}"
        if setup.distribute_strategy == "mirrored":
            cfg_str += "-mirrored"

        path = path / Path(cfg_str.format(rho=setup.rho, alpha=setup.alpha, beta=setup.beta,
                                          lambda_weight=setup.lambda_weight))

    elif model_type == "CASTLEAdapted":
        cfg_str = "r{rho}-a{alpha}-lpred{lambda_prediction}-lspar{lambda_sparsity}-" \
                  "lrec{lambda_reconstruction}-lacy{lambda_acyclicity}-{acyclicity_constraint}"
        if setup.distribute_strategy == "mirrored":
            cfg_str += "-mirrored"

        path = path / Path(cfg_str.format(rho=setup.rho, alpha=setup.alpha,
                                          lambda_prediction=setup.lambda_prediction,
                                          lambda_sparsity=setup.lambda_sparsity,
                                          lambda_reconstruction=setup.lambda_reconstruction,
                                          lambda_acyclicity=setup.lambda_acyclicity,
                                          acyclicity_constraint=setup.acyclicity_constraint))
    elif model_type == "GumbelSoftmaxSingleOutputModel":
        cfg_str = "lspar{lambda_sparsity}"
        if setup.distribute_strategy == "mirrored":
            cfg_str += "-mirrored"

        path = path / Path(cfg_str.format(lambda_sparsity=setup.lambda_sparsity))

    elif model_type == "CASTLESimplified":
        # Legacy version of GumbelSoftmaxSingleOutputModel for backwards compatibility
        cfg_str = "lspar{lambda_sparsity}"
        if setup.distribute_strategy == "mirrored":
            cfg_str += "-mirrored"

        path = path / Path(cfg_str.format(lambda_sparsity=setup.lambda_sparsity))

    elif model_type == "castleNN":
        # Legacy version of CASTLE for backwards compatibility
        if setup.distribute_strategy == "mirrored":
            cfg_str = "r{rho}-a{alpha}-b{beta}-l{lambda_weight}-mirrored/"
        elif setup.distribute_strategy == "multi_worker_mirrored":
            cfg_str = "r{rho}-a{alpha}-b{beta}-l{lambda_weight}-multi_worker_mirrored/"
        else:
            cfg_str = "r{rho}-a{alpha}-b{beta}-l{lambda_weight}/"
        path = path / Path(cfg_str.format(rho=setup.rho, alpha=setup.alpha, beta=setup.beta,
                                          lambda_weight=setup.lambda_weight))

    str_hl = str(setup.hidden_layers).replace(", ", "_")
    str_hl = str_hl.replace("[", "").replace("]", "")
    str_act = str(setup.activation)

    training_castle = model_type in ["CASTLEOriginal", "CASTLEAdapted", "CASTLESimplified",
                                     "GumbelSoftmaxSingleOutputModel"]
    if str_act.lower() == "leakyrelu" and training_castle:
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
    """ Generate a filename to save the model """
    i_var = setup.output_order.index(output.var)
    i_level = output.level_idx
    if i_level is None:
        i_level = 0
    return f"{i_var}_{i_level}"


def get_best_ckpt_path(setup, model_type, output, pc_alpha=None, threshold=None):
    folder = get_path(setup, model_type, pc_alpha=pc_alpha, threshold=threshold)
    return Path(folder, "ckpt_best", get_filename(setup, output) + "_model", "best_train_ckpt")


def get_cont_ckpt_path(setup, model_type, output, pc_alpha=None, threshold=None):
    folder = get_path(setup, model_type, pc_alpha=pc_alpha, threshold=threshold)
    return Path(folder, "ckpt_cont", get_filename(setup, output) + "_model", "cont_train_ckpt")


def load_model_weights_from_checkpoint(model_description, which_checkpoint):
    if which_checkpoint == "best":
        ckpt_path = get_best_ckpt_path(model_description.setup, model_description.model_type, model_description.output,
                                       pc_alpha=model_description.pc_alpha, threshold=model_description.threshold)

    elif which_checkpoint == "cont":
        ckpt_path = get_cont_ckpt_path(model_description.setup, model_description.model_type, model_description.output,
                                       pc_alpha=model_description.pc_alpha, threshold=model_description.threshold)

    else:
        raise ValueError(f"Which checkpoint value must be in ['best', 'cont']")

    print(f"\nLoading model weights from checkpoint path {ckpt_path}.\n")
    model_description.model.load_weights(ckpt_path)

    return model_description


def load_model_from_previous_training(model_description):
    model, _ = get_model(model_description.setup, model_description.output, model_description.model_type,
                         pc_alpha=model_description.pc_alpha, threshold=model_description.threshold)
    return model


def get_model(setup, output, model_type, *, pc_alpha=None, threshold=None):
    """ Get model and input list """
    folder = get_path(setup, model_type, pc_alpha=pc_alpha, threshold=threshold)
    filename = get_filename(setup, output)

    if setup.nn_type == "CASTLEOriginal":
        modelname = Path(folder, filename + '_model.keras')
        print(f"\nLoad model: {modelname}")

        model = tf.keras.models.load_model(modelname, custom_objects={'CASTLEOriginal': CASTLEOriginal,
                                                                      'MaskedDenseLayers': MaskedDenseLayer})
    elif setup.nn_type == "CASTLEAdapted":
        modelname = Path(folder, filename + '_model.keras')
        print(f"\nLoad model: {modelname}")

        model = tf.keras.models.load_model(modelname, custom_objects={'CASTLEAdapted': CASTLEAdapted,
                                                                      'MaskedDenseLayers': MaskedDenseLayer})
    elif setup.nn_type == "CASTLESimplified":
        modelname = Path(folder, filename + '_model.keras')
        print(f"\nLoad model: {modelname}")

        model = tf.keras.models.load_model(modelname, custom_objects={'CASTLESimplified': CASTLESimplified})

    elif setup.nn_type == "GumbelSoftmaxSingleOutputModel":
        modelname = Path(folder, filename + '_model.keras')
        print(f"\nLoad model: {modelname}")

        model = tf.keras.models.load_model(modelname, custom_objects={
            'GumbelSoftmaxSingleOutputModel': GumbelSoftmaxSingleOutputModel,
            'StraightThroughGumbelSoftmaxMaskingLayer': StraightThroughGumbelSoftmaxMaskingLayer})


    elif setup.nn_type == "castleNN":
        # Legacy version of CASTLE for backwards compatibility
        modelname = Path(folder, filename + '_model.keras')
        print(f"\nLoad model: {modelname}")

        model = tf.keras.models.load_model(modelname, custom_objects={'CASTLE': CASTLE,
                                                                      'MaskedDenseLayers': MaskedDenseLayer})

    else:
        modelname = Path(folder, filename + '_model.h5')
        print(f"\nLoad model: {modelname}")

        model = tf.keras.models.load_model(modelname)

    inputs_path = Path(folder, f"{filename}_input_list.txt")
    with open(inputs_path) as inputs_file:
        input_indices = [i for i, v in enumerate(inputs_file.readlines()) if int(v)]

    return (model, input_indices)


def get_var_list(setup, target_vars):
    output_list = list()
    for spcam_var in target_vars:
        if spcam_var.dimensions == 3:
            var_levels = [setup.children_idx_levs, setup.parents_idx_levs] \
                [spcam_var.type == 'in']
            for level, _ in var_levels:
                # There's enough info to build a Variable_Lev_Metadata list
                # However, it could be better to do a bigger reorganization
                var_name = f"{spcam_var.name}-{round(level, 2)}"
                output_list.append(var_name)
        elif spcam_var.dimensions == 2:
            var_name = spcam_var.name
            output_list.append(var_name)
    return output_list


def load_single_model(setup, var_name):
    loading_castle = setup.nn_type == "CASTLEOriginal" or setup.nn_type == "CASTLEAdapted" or \
                     setup.nn_type == "CASTLESimplified" or setup.nn_type == "GumbelSoftmaxSingleOutputModel" or \
                     setup.nn_type == "castleNN"
    if setup.do_single_nn or setup.do_random_single_nn or setup.do_pca_nn or setup.do_sklasso_nn or loading_castle:
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
    """ Load all NN models specified in setup """
    models = collections.defaultdict(dict)

    output_list = get_var_list(setup, setup.spcam_outputs)
    model_is_castle = setup.nn_type in ["CASTLEOriginal", "CASTLEAdapted", "GumbelSoftmaxSingleOutputModel",
                                        "CASTLESimplified", "castleNN"]
    if setup.do_single_nn or setup.do_random_single_nn or setup.do_pca_nn or setup.do_sklasso_nn or model_is_castle:
        nn_type = setup.nn_type  # if setup.do_random_single_nn else 'SingleNN'
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
                    # todo: remove if not necessary
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
    folder = get_path(setup, model_type, pc_alpha=pc_alpha, threshold=threshold)
    path = Path(folder, 'diagnostics')
    return path
