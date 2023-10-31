import sys

import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Activation

from utils.constants import SPCAM_Vars
from utils.variable import Variable_Lev_Metadata
import utils.pcmci_aggregation as aggregation
from neural_networks.sklearn_lasso import sklasso
from neural_networks.castle.building_castle import build_castle


class ModelDescription:
    """ Object that stores a Keras model and metainformation about it.

    Attributes
    ----------
    output : Variable_Lev_Metadata
        Output variable of the model.
    pc_alpha : str
        Meta information. PC alpha used to find the inputs.
    threshold : str
        Meta information. Gridpoint threshold used to select the inputs.
    inputs : list(Variable)
        List of the variables (and variable level) that cause the output
        variable.
    hidden_layers : list(int)
        Description of the hidden dense layers of the model
        (default [32, 32, 32]).
    activation : Keras-compatible activation function
        Activation function used for the hidden dense layers
        (default "relu").
    model : Keras model
        Model created using the given information.
        See `_build_model()`.
    input_vars_dict:
    output_vars_dict:

    #TODO

    """

    def __init__(self, output, inputs, model_type, pc_alpha, threshold, setup, tuning_params):
        """
        Parameters
        ----------
        output : str
            Output variable of the model in string format. See Variable_Lev_Metadata.
        inputs : list(str)
            List of strings for the variables that cause the output variable.
            See Variable_Lev_Metadata.
        model_type : str
            # TODO
        pc_alpha : str
            Meta information. PC alpha used to find the inputs.
        threshold : str
            Meta information. Gridpoint threshold used to select the inputs.
        hidden_layers : list(int)
            Description of the hidden dense layers of the model.
        activation : Keras-compatible activation function
            Activation function used for the hidden dense layers.
        """
        self.setup = setup
        self.output = Variable_Lev_Metadata.parse_var_name(output)
        self.inputs = sorted(
            [Variable_Lev_Metadata.parse_var_name(p) for p in inputs],
            key=lambda x: self.setup.input_order_list.index(x),
        )

        self.model_type = model_type
        self.pc_alpha = pc_alpha
        self.threshold = threshold
        if hasattr(setup, 'sherpa_hyper'):
            self.setup.hidden_layers = [setup.num_nodes] * setup.num_layers
            # TODO: setup activation coefficient

        self.input_vars_dict = ModelDescription._build_vars_dict(self.inputs)
        self.output_vars_dict = ModelDescription._build_vars_dict([self.output])

        setup.input_pca_vars_dict = self.input_pca_vars_dict = False
        if setup.do_pca_nn:
            setup.inputs_pca = self.inputs_pca = self.inputs[:int(setup.n_components)]
            setup.input_pca_vars_dict = self.input_pca_vars_dict = ModelDescription._build_vars_dict(setup.inputs_pca)

        if setup.do_sklasso_nn: self.lasso_coefs = setup.lasso_coefs

        hidden_layers = [tuning_params['dense_units']] * tuning_params["num_hidden_layers"]
        learning_rate = tuning_params['learning_rate']
        activation = tuning_params['activation_type']
        lambda_weight = tf.cast(tuning_params['lambda_weight'], dtype=tf.float32)

        if setup.nn_type == "CASTLEOriginal" or setup.nn_type == "CASTLEAdapted":
            if setup.distribute_strategy == "mirrored":
                # Train with MirroredStrategy across multiple GPUs
                self.strategy = tf.distribute.MirroredStrategy()
            elif setup.distribute_strategy == "multi_worker_mirrored":
                # Train with MultiWorkerMirrored strategy across multiple SLURM nodes following
                #   http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-tf-multi-eng.html
                # Build multi-worker environment from Slurm variables
                cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=33001)
                print(f"\n\nCluster resolver cluster spec: \n{cluster_resolver.cluster_spec()}\n\n")
                print(f"\n\nCluster resolver cluster spec: \n{cluster_resolver.get_task_info()}\n\n")

                # Use NCCL communication protocol
                implementation = tf.distribute.experimental.CommunicationImplementation.AUTO
                communication_options = tf.distribute.experimental.CommunicationOptions(implementation=implementation)

                # Declare distribution strategy
                self.strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=cluster_resolver,
                                                                          communication_options=communication_options)
            else:
                self.strategy = None

            self.model = build_castle(num_x_inputs=len(self.inputs),
                                      hidden_layers=hidden_layers,
                                      activation=activation, rho=self.setup.rho, alpha=self.setup.alpha,
                                      lambda_weight=lambda_weight, learning_rate=learning_rate, strategy=self.strategy)
        else:
            self.model = self._build_model(learning_rate)

    def _build_model(self, hidden_layers, activation, learning_rate):
        """ Build a Keras model with the given information.

        Some parameters are not configurable, taken from Rasp et al.
        """
        input_shape = len(self.inputs)
        if self.model_type == "pcaNN": input_shape = len(self.inputs_pca)
        input_shape = (input_shape,)
        model = dense_nn(
            input_shape=input_shape,
            output_shape=1,  # Only one output per model
            hidden_layers=hidden_layers,
            activation=activation
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam",
        )

        #         optimizer = tf.keras.optimizers.RMSprop(
        #             learning_rate=0.001,
        #             rho=0.9,
        #             momentum=0.0,
        #             epsilon=1e-07,
        #             centered=False,
        #             name="RMSprop",
        #         )

        model.compile(
            # TODO? Move to configuration
            optimizer=optimizer,
            loss="mse",  # From 006_8col_pnas_exact.yml
            metrics=[tf.keras.losses.mse],  # From train.py (default)
        )
        return model

    @staticmethod
    def _build_vars_dict(list_variables):
        """ Convert the given list of Variable_Lev_Metadata into a
        dictionary to be used on the data generator.

        Parameters
        ----------
        list_variables : list(Variable_Lev_Metadata)
            List of variables to be converted to the dictionary format
            used by the data generator

        Returns
        -------
        vars_dict : dict{str : list(int)}
            Dictionary of the form {ds_name : list of levels}, where
            "ds_name" is the name of the variable as stored in the
            dataset, and "list of levels" a list containing the indices
            of the levels of that variable to use, or None for 2D
            variables.
        """
        vars_dict = dict()
        for variable in list_variables:
            ds_name = variable.var.ds_name  # Name used in the dataset
            if variable.var.dimensions == 2:
                ds_name = 'LHF_nsDELQ' if ds_name == 'LHF_NSDELQ' else ds_name
                vars_dict[ds_name] = None
            elif variable.var.dimensions == 3:
                levels = vars_dict.get(ds_name, list())
                levels.append(variable.level_idx)
                vars_dict[ds_name] = levels
        return vars_dict

    def fit_model(self, x, validation_data, epochs, callbacks, verbose=1, steps_per_epoch=None, validation_steps=None):
        """ Train the model """
        history = self.model.fit(
            x=x,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
        )

        return history

    def get_path(self, base_path):
        """ Generate a path based on this model metadata """
        path = Path(base_path, self.model_type)
        if self.model_type == "CausalSingleNN" or self.model_type == "CorrSingleNN":
            if self.setup.area_weighted:
                cfg_str = "a{pc_alpha}-t{threshold}-latwts/"
            else:
                cfg_str = "a{pc_alpha}-t{threshold}/"
            path = path / Path(
                cfg_str.format(pc_alpha=self.pc_alpha, threshold=self.threshold)
            )
        elif self.model_type == "pcaNN":
            cfg_str = "pcs{n_components}/"
            path = path / Path(
                cfg_str.format(n_components=self.setup.n_components)
            )
        elif "lasso" in self.model_type:
            cfg_str = "a{alpha_lasso}/"
            path = path / Path(
                cfg_str.format(alpha_lasso=self.setup.alpha_lasso)
            )
        elif self.model_type == "castleNN":
            if self.setup.distribute_strategy == "mirrored":
                cfg_str = "r{rho}-a{alpha}-b{beta}-l{lambda_weight}-mirrored/"
            elif self.setup.distribute_strategy == "multi_worker_mirrored":
                cfg_str = "r{rho}-a{alpha}-b{beta}-l{lambda_weight}-multi_worker_mirrored/"
            else:
                cfg_str = "r{rho}-a{alpha}-b{beta}-l{lambda_weight}/"
            path = path / Path(cfg_str.format(rho=self.setup.rho, alpha=self.setup.alpha, beta=self.setup.beta,
                                              lambda_weight=self.setup.lambda_weight))

        str_hl = str(self.setup.hidden_layers).replace(", ", "_")
        str_hl = str_hl.replace("[", "").replace("]", "")
        path = path / Path(
            "hl_{hidden_layers}-act_{activation}-e_{epochs}/".format(
                hidden_layers=str_hl,
                activation=self.setup.activation,
                epochs=self.setup.epochs,
            )
        )
        return path

    def get_filename(self):
        """ Generate a filename to save the model """
        i_var = self.setup.output_order.index(self.output.var)
        i_level = self.output.level_idx
        if i_level is None:
            i_level = 0
        return f"{i_var}_{i_level}"

    def save_model(self, base_path):
        """ Save model, weights and input list """
        folder = self.get_path(base_path)
        filename = self.get_filename()
        print(f"\nUsing filename {filename}.\n")
        # Save model
        Path(folder).mkdir(parents=True, exist_ok=True)

        if self.setup.nn_type == "CASTLEOriginal" or self.setup.nn_type == "CASTLEAdapted":
            # Castle model is custom, so it cannot be saved in legacy h5 format
            self.model.save(Path(folder, f"{filename}_model.keras"), save_format="keras_v3")
        else:
            self.model.save(Path(folder, f"{filename}_model.h5"))
        # Save weights
        self.model.save_weights(str(Path(folder, f"{filename}_weights.h5")))
        # Save input list
        self.save_input_list(folder, filename)

    def save_input_list(self, folder, filename):
        """ Save input list """
        input_list = self.get_input_list()
        with open(Path(folder, f"{filename}_input_list.txt"), "w") as f:
            for line in input_list:
                print(str(line), file=f)

    def get_input_list(self):
        """ Generate input list """
        input_list = self.inputs
        if self.model_type == "pcaNN": input_list = self.inputs_pca
        return [int(var in input_list) for var in self.setup.input_order_list]

    def __str__(self):
        name = f"{self.model_type}: {self.output}"
        if self.pc_alpha != None:
            # pc_alpha and threshold should be either both None or both not None
            name += f", a{self.pc_alpha}-t{self.threshold}"
        return name

    def __repr__(self):
        return repr(str(self))

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


def dense_nn(input_shape, output_shape, hidden_layers, activation):
    """ Create a dense NN in base of the parameters received """
    model = Sequential()
    model.add(Input(shape=input_shape))

    for n_layer_nodes in hidden_layers:
        act = tf.keras.layers.LeakyReLU(alpha=0.3) if activation == 'LeakyReLU' else Activation(activation)
        #         act = tf.keras.layers.LeakyReLU(alpha=0.3) if activation=='LeakyReLU' else activation
        #         model.add(Dense(n_layer_nodes, activation=act))
        model.add(Dense(n_layer_nodes))
        model.add(act)

    model.add(Dense(output_shape))
    return model


def generate_all_single_nn(setup, tuning_params):
    """
    SingleNN: Generate all NN with one output and all inputs specified in the setup
    pcaNN:    Generate all NN with one output and PCs (PCA) as inputs
    castleNN: Generate all NN with one output and all inputs specified in the setup
    """
    model_descriptions = list()

    inputs = list()  # TODO Parents and levels
    for spcam_var in setup.spcam_inputs:
        if spcam_var.dimensions == 3:
            for level, _ in setup.parents_idx_levs:
                # There's enough info to build a Variable_Lev_Metadata list
                # However, it could be better to do a bigger reorganization
                var_name = f"{spcam_var.name}-{round(level, 2)}"
                inputs.append(var_name)
        elif spcam_var.dimensions == 2:
            var_name = spcam_var.name
            inputs.append(var_name)

    output_list = list()
    for spcam_var in setup.spcam_outputs:
        if spcam_var.dimensions == 3:
            for level, _ in setup.children_idx_levs:
                # There's enough info to build a Variable_Lev_Metadata list
                # However, it could be better to do a bigger reorganization
                var_name = f"{spcam_var.name}-{round(level, 2)}"
                output_list.append(var_name)
        elif spcam_var.dimensions == 2:
            var_name = spcam_var.name
            output_list.append(var_name)

    for output in output_list:
        model_description = ModelDescription(
            output, inputs, setup.nn_type, pc_alpha=None, threshold=None, setup=setup, tuning_params=tuning_params
        )
        model_descriptions.append(model_description)
    return model_descriptions


def generate_all_causal_single_nn(setup, aggregated_results):
    """ Generate all NN with one output and selected inputs from a pc analysis """

    model_descriptions = list()

    for output, pc_alpha_dict in aggregated_results.items():
        print(output)
        if len(pc_alpha_dict) == 0:  # May be empty
            # TODO How to approach this?
            print("Empty results")
            pass
        for pc_alpha, pc_alpha_results in pc_alpha_dict.items():
            var_names = np.array(pc_alpha_results["var_names"])
            for threshold, parent_idxs in pc_alpha_results["parents"].items():
                parents = var_names[parent_idxs]
                # print(f"output: {output}")
                # print(f"parents: {parents}")
                # print(f"var_names[parent_idxs]: {var_names[parent_idxs]}")
                # print(f"parent_idxs: {parent_idxs}")
                model_description = ModelDescription(
                    output, parents, setup.nn_type, pc_alpha, threshold, setup=setup,
                )
                model_descriptions.append(model_description)
    return model_descriptions


def generate_sklasso_single_nn(setup):
    """
    sklassoNN: Generate all NN with one output and Lasso (L1) as inputs
    """
    model_descriptions = list()

    inputs = list()
    for spcam_var in setup.spcam_inputs:
        if spcam_var.dimensions == 3:
            for level, _ in setup.parents_idx_levs:
                # There's enough info to build a Variable_Lev_Metadata list
                # However, it could be better to do a bigger reorganization
                var_name = f"{spcam_var.name}-{round(level, 2)}"
                inputs.append(var_name)
        elif spcam_var.dimensions == 2:
            var_name = spcam_var.name
            inputs.append(var_name)
    inputs = sorted(
        [Variable_Lev_Metadata.parse_var_name(p) for p in inputs],
        key=lambda x: setup.input_order_list.index(x),
    )

    output_list = list()
    for spcam_var in setup.spcam_outputs:
        if spcam_var.dimensions == 3:
            for level, _ in setup.children_idx_levs:
                # There's enough info to build a Variable_Lev_Metadata list
                # However, it could be better to do a bigger reorganization
                var_name = f"{spcam_var.name}-{round(level, 2)}"
                output_list.append(var_name)
        elif spcam_var.dimensions == 2:
            var_name = spcam_var.name
        output_list.append(var_name)

    for output in output_list:
        output = Variable_Lev_Metadata.parse_var_name(output)
        lasso_inputs, lasso_coefs = sklasso(
            inputs=inputs,
            output=output,
            data_fn=Path(setup.train_data_folder, setup.train_data_fn),
            norm_fn=Path(setup.normalization_folder, setup.normalization_fn),
            setup=setup,
        )
        setup.lasso_coefs = lasso_coefs
        print(f"lasso_inputs: {lasso_inputs}")
        model_description = ModelDescription(
            output, lasso_inputs, setup.nn_type, pc_alpha=None, threshold=None, setup=setup,
        )
        model_descriptions.append(model_description)
    return model_descriptions


def generate_models(setup, tuning_params, threshold_dict=False):
    """ Generate all NN models specified in setup """
    model_descriptions = list()

    if setup.distribute_strategy == "mirrored" or setup.distribute_strategy == "multi_worker_mirrored":
        if not tf.config.get_visible_devices('GPU'):
            raise EnvironmentError(f"Cannot build and compile models with tf.distribute.MirroredStrategy "
                                   f"because Tensorflow found no GPUs.")
        print(f"\n\nBuilding and compiling models with tf.distribute.MirroredStrategy.", flush=True)
    else:
        print(f"\n\nBuilding and compiling models.", flush=True)

    if setup.do_single_nn or setup.do_pca_nn or setup.nn_type == "CASTLEOriginal" or setup.nn_type == "CASTLEAdapted":
        model_descriptions.extend(generate_all_single_nn(setup, tuning_params))

    if setup.do_random_single_nn:
        collected_results, errors = aggregation.collect_results(setup, reuse=True)
        aggregation.print_errors(errors)

        aggregated_results, var_names_parents = aggregation.aggregate_results_for_numparents(
            collected_results, setup, thresholds_dict=setup.thrs_argv, random=setup.random
        )
        # aggregated_results, var_names_parents = aggregation.aggregate_results_for_numparents(
        #     collected_results, setup.numparents_argv, setup, random=setup.random
        # )
        model_descriptions.extend(
            generate_all_causal_single_nn(setup, aggregated_results)
        )

    if setup.do_causal_single_nn:
        collected_results, errors = aggregation.collect_results(setup, reuse=True)
        aggregation.print_errors(errors)
        aggregated_results, var_names_parents = aggregation.aggregate_results(
            collected_results, setup, threshold_dict=threshold_dict)
        model_descriptions.extend(
            generate_all_causal_single_nn(setup, aggregated_results)
        )

    if setup.do_sklasso_nn:
        model_descriptions.extend(generate_sklasso_single_nn(setup))

    return model_descriptions


def generate_model_sherpa(setup, parents=False, pc_alpha=None, threshold=None):
    """ Generate NN model for hyperparameter tuning via Sherpa """

    if setup.do_causal_single_nn and parents == False:
        parents = get_parents_sherpa(setup)

    model_description = ModelDescription(
        setup.output, parents, setup.nn_type, pc_alpha, threshold, setup=setup,
    )
    return model_description


def generate_input_list(setup):
    """ Generate input list for hyperparameter tuning via Sherpa """
    inputs = list()
    for spcam_var in setup.spcam_inputs:
        if spcam_var.dimensions == 3:
            for level, _ in setup.parents_idx_levs:
                var_name = f"{spcam_var.name}-{round(level, 2)}"
                inputs.append(var_name)
        elif spcam_var.dimensions == 2:
            var_name = spcam_var.name
            inputs.append(var_name)
    return inputs


def generate_output_list(setup):
    """ Generate output list for hyperparameter tuning via Sherpa """
    output_list = list()
    for spcam_var in setup.spcam_outputs:
        if spcam_var.dimensions == 3:
            for level, _ in setup.children_idx_levs:
                var_name = f"{spcam_var.name}-{round(level, 2)}"
                output_list.append(var_name)
        elif spcam_var.dimensions == 2:
            var_name = spcam_var.name
            output_list.append(var_name)
    return output_list


def get_parents_sherpa(setup):
    collected_results, errors = aggregation.collect_results(setup, reuse=True)
    aggregation.print_errors(errors)
    aggregated_results, var_names_parents = aggregation.aggregate_results(
        collected_results, setup
    )
    output = list(aggregated_results.keys())[0]
    if str(setup.output) == str(output):
        pass
    else:
        print(f"output from output_list and output from aggregated results do not match; stop")
        import pdb;
        pdb.set_trace()
    pc_alpha = list(aggregated_results[output].keys())[0]
    threshold = list(aggregated_results[output][pc_alpha]['parents'].keys())[0]
    var_names = np.array(aggregated_results[output][pc_alpha]['var_names'])
    parent_idxs = aggregated_results[output][pc_alpha]['parents'][threshold]
    parents = var_names[parent_idxs]
    return parents, pc_alpha, threshold
