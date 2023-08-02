import collections
from pathlib import Path

import tensorflow as tf

from utils.variable import Variable_Lev_Metadata
from neural_networks.castle_model import CASTLE


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
    elif model_type == "castleNN":
        if setup.do_mirrored_strategy:
            cfg_str = "r{rho}-a{alpha}-b{beta}-l{lambda_}-distributed/"
        else:
            cfg_str = "r{rho}-a{alpha}-b{beta}-l{lambda_}/"
        path = path / Path(cfg_str.format(rho=setup.rho, alpha=setup.alpha, beta=setup.beta, lambda_=setup.lambda_))

    str_hl = str(setup.hidden_layers).replace(", ", "_")
    str_hl = str_hl.replace("[", "").replace("]", "")
    path = path / Path(
        "hl_{hidden_layers}-act_{activation}-e_{epochs}/".format(
            hidden_layers=str_hl,
            activation=setup.activation,
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


def get_model(setup, output, model_type, *, pc_alpha=None, threshold=None):
    """ Get model and input list """
    folder = get_path(setup, model_type, pc_alpha=pc_alpha, threshold=threshold)
    filename = get_filename(setup, output)

    if setup.do_castle_nn:
        modelname = Path(folder, filename + '_model.keras')
        print(f"\nLoad model: {modelname}")

        model = tf.keras.models.load_model(modelname, custom_objects={'CASTLE': CASTLE})
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
    if setup.do_single_nn or setup.do_random_single_nn or setup.do_pca_nn or setup.do_sklasso_nn or setup.do_castle_nn:
        var = Variable_Lev_Metadata.parse_var_name(var_name)
        return {var: get_model(setup, var, setup.nn_type, pc_alpha=None, threshold=None)}
    else:
        raise NotImplementedError(f"load_single_model is not implemented for neural network type {setup.nn_type}")


def load_models(setup):
    """ Load all NN models specified in setup """
    models = collections.defaultdict(dict)

    output_list = get_var_list(setup, setup.spcam_outputs)
    if setup.do_single_nn or setup.do_random_single_nn or setup.do_pca_nn or setup.do_sklasso_nn or setup.do_castle_nn:
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
