import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Activation

from pcmasking import utils as aggregation
from pcmasking.neural_networks.custom_models.building_custom_model import build_custom_model
from pcmasking.neural_networks.load_models import get_mask_net_threshold
from pcmasking.utils.variable import Variable_Lev_Metadata


class ModelDescription:
    """A class to store a Keras model and its associated metadata.

    Attributes:
        output (Variable_Lev_Metadata): Output variable of the model.
        pc_alpha (str): Regularization parameter for PC1.
        threshold (str): Causal threshold for PC1.
        inputs (list): List of input variables.
        hidden_layers (list): List of hidden layers in the model.
        activation (str): Activation function for hidden layers.
        model (keras.Model): The built Keras model.
        input_vars_dict (dict): Dictionary of input variables and their levels.
        output_vars_dict (dict): Dictionary of output variables and their levels.
    """

    def __init__(self, output, inputs, model_type, pc_alpha, threshold, setup, continue_training=False, seed=None):
        """Initializes the ModelDescription object with the given parameters.

        Args:
            output (str): Output variable name.
            inputs (list): List of input variables.
            model_type (str): Type of model to build.
            pc_alpha (str): Regularization parameter for PC1.
            threshold (str): Causal threshold for PC1.
            setup (pcmasking.utils.setup.Setup): Setup object with configuration details.
            continue_training (bool, optional): Whether to continue training from a previous checkpoint.
                Defaults to False.
            seed (int, optional): Seed for reproducibility. Defaults to None.
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

        self.input_vars_dict = ModelDescription._build_vars_dict(self.inputs)
        self.output_vars_dict = ModelDescription._build_vars_dict([self.output])

        self.seed = seed

        training_pcmasking = self.model_type in ["PreMaskNet", "MaskNet"]
        if training_pcmasking:
            if setup.distribute_strategy == "mirrored":
                # Train with MirroredStrategy across multiple GPUs
                self.strategy = tf.distribute.MirroredStrategy()
            else:
                self.strategy = None

            learning_rate = setup.init_lr

            if continue_training:
                save_dir = str(self.get_path(setup.nn_output_path))

                previous_lr_path = Path(save_dir, "learning_rate", self.get_filename() + "_model_lr.p")
                print(f"\nLoading learning rate from {previous_lr_path}", flush=True)

                with open(previous_lr_path, 'rb') as f:
                    learning_rate = pickle.load(f)["last_lr"]
                print(f"Learning rate = {learning_rate}\n", flush=True)

            self.model = build_custom_model(self.setup, num_x_inputs=len(self.inputs), learning_rate=learning_rate,
                                            output_var=self.output, strategy=self.strategy, seed=self.seed)
        else:
            self.model = self._build_model()

    def _build_model(self):
        """Builds a dense neural network model based on the specified setup configuration.
        Some parameters are not configurable, taken from Rasp et al.

        Returns:
            keras.Model: Compiled Keras model.
        """
        input_shape = len(self.inputs)

        input_shape = (input_shape,)
        model = dense_nn(
            input_shape=input_shape,
            output_shape=1,  # Only one output per model
            hidden_layers=self.setup.hidden_layers,
            activation=self.setup.activation,
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam",
        )

        model.compile(
            optimizer=optimizer,
            loss="mse",  # From 006_8col_pnas_exact.yml
            metrics=[tf.keras.losses.mse],  # From train.py (default)
        )
        return model

    @staticmethod
    def _build_vars_dict(list_variables):
        """
        Converts a list of Variable_Lev_Metadata into a dictionary used by the data generator.

        Args:
            list_variables (list): List of Variable_Lev_Metadata.

        Returns:
            dict: Dictionary with dataset names as keys and levels as values.
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
        """Trains the model using the provided training data.

        Args:
            x (tuple): Training data (inputs and labels).
            validation_data (tuple): Validation data (inputs and labels).
            epochs (int): Number of training epochs.
            callbacks (list): List of Keras callbacks.
            verbose (int, optional): Verbosity level. Defaults to 1.
            steps_per_epoch (int, optional): Number of steps per epoch. Defaults to None.
            validation_steps (int, optional): Number of validation steps. Defaults to None.

        Returns:
            History: Keras training history object.
        """
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
        """ Generates a path based on the model's metadata for saving or loading.

        Args:
            base_path (str): Base path for saving or loading the model.

        Returns:
            Path: A pathlib Path object representing the model path.
        """
        path = Path(base_path, self.model_type)

        if self.model_type == "CausalSingleNN" or self.model_type == "CorrSingleNN":
            if self.setup.area_weighted:
                cfg_str = "a{pc_alpha}-t{threshold}-latwts/"
            else:
                cfg_str = "a{pc_alpha}-t{threshold}/"
            path = path / Path(
                cfg_str.format(pc_alpha=self.pc_alpha, threshold=self.threshold)
            )

        elif self.model_type == "MaskNet":
            cfg_str = "threshold{threshold}"

            if self.setup.distribute_strategy == "mirrored":
                cfg_str += "-mirrored"

            if self.setup.mask_threshold is None:
                self.setup.mask_threshold = get_mask_net_threshold(self.setup, self.output)
                path = path / Path(cfg_str.format(threshold=self.setup.mask_threshold))

                # Reset threshold
                self.setup.mask_threshold = None
            else:
                path = path / Path(cfg_str.format(threshold=self.setup.mask_threshold))

        elif self.model_type == "PreMaskNet":
            cfg_str = "lspar{lambda_sparsity}"
            if self.setup.distribute_strategy == "mirrored":
                cfg_str += "-mirrored"

            path = path / Path(cfg_str.format(lambda_sparsity=self.setup.lambda_sparsity))

        str_hl = str(self.setup.hidden_layers).replace(", ", "_")
        str_hl = str_hl.replace("[", "").replace("]", "")
        str_act = str(self.setup.activation)
        training_pcmasking = self.model_type in ["PreMaskNet", "MaskNet"]
        if str_act.lower() == "leakyrelu" and training_pcmasking:
            str_act += f"_{self.setup.relu_alpha}"

        path = path / Path(
            "hl_{hidden_layers}-act_{activation}-e_{epochs}/".format(
                hidden_layers=str_hl,
                activation=str_act,
                epochs=self.setup.epochs,
            )
        )
        return path

    def get_filename(self):
        """Generates a filename based on the model's output variable and level """
        """ Generate a filename to save the model """
        i_var = self.setup.output_order.index(self.output.var)
        i_level = self.output.level_idx
        if i_level is None:
            i_level = 0
        return f"{i_var}_{i_level}"

    def save_model(self, base_path):
        """Saves the model, weights, and input list to the specified directory"""
        folder = self.get_path(base_path)
        filename = self.get_filename()
        print(f"\nUsing filename {filename}.\n")
        # Save model
        Path(folder).mkdir(parents=True, exist_ok=True)

        training_pcmasking = self.model_type in ["PreMaskNet", "MaskNet"]
        if training_pcmasking:
            # Custom models are saved in new keras format
            self.model.save(Path(folder, f"{filename}_model.keras"), save_format="keras_v3")
        else:
            self.model.save(Path(folder, f"{filename}_model.h5"))
        # Save weights
        self.model.save_weights(str(Path(folder, f"{filename}_weights.h5")))
        # Save input list
        self.save_input_list(folder, filename)

    def save_input_list(self, folder, filename):
        """Saves the list of input variables used by the model to a text file """
        input_list = self.get_input_list()
        with open(Path(folder, f"{filename}_input_list.txt"), "w") as f:
            for line in input_list:
                print(str(line), file=f)

    def get_input_list(self):
        """Generates an ordered list of all inputs"""
        input_list = self.inputs
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
    """ Builds a dense neural network model.

    Args:
        input_shape (tuple): Shape of the input data.
        output_shape (int): Number of output units.
        hidden_layers (list): List of integers specifying the number of units in each hidden layer.
        activation (str): Activation function to use for the hidden layers.

    Returns:
        keras.Model: A Keras Sequential model with the specified architecture.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    for n_layer_nodes in hidden_layers:
        act = tf.keras.layers.LeakyReLU(alpha=0.3) if activation == 'LeakyReLU' else Activation(activation)

        model.add(Dense(n_layer_nodes))
        model.add(act)

    model.add(Dense(output_shape))
    return model


def generate_all_single_nn(setup, continue_training=False, seed=None):
    """
    Generates all neural networks (SingleNN, PreMaskNet, MaskNet) with one output and all inputs
    specified in the setup.

    Args:
        setup (object): Setup object containing configuration and input/output variables.
        continue_training (bool, optional): Whether to continue training from previous checkpoints. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        list: A list of ModelDescription objects for each single-output neural network.
    """
    model_descriptions = list()

    inputs = list()
    for spcam_var in setup.spcam_inputs:
        if spcam_var.dimensions == 3:
            for level, _ in setup.parents_idx_levs:
                var_name = f"{spcam_var.name}-{round(level, 2)}"
                inputs.append(var_name)
        elif spcam_var.dimensions == 2:
            var_name = spcam_var.name
            inputs.append(var_name)

    output_list = list()
    for spcam_var in setup.spcam_outputs:
        if spcam_var.dimensions == 3:
            for level, _ in setup.children_idx_levs:
                var_name = f"{spcam_var.name}-{round(level, 2)}"
                output_list.append(var_name)
        elif spcam_var.dimensions == 2:
            var_name = spcam_var.name
            output_list.append(var_name)

    for output in output_list:
        model_description = ModelDescription(
            output, inputs, setup.nn_type, pc_alpha=None, threshold=None, setup=setup,
            continue_training=continue_training, seed=seed
        )
        model_descriptions.append(model_description)
    return model_descriptions


def generate_all_causal_single_nn(setup, aggregated_results, seed=None):
    """
    Generates all neural networks with one output and selected inputs from PC1 analysis.

    Args:
        setup (object): Setup object containing configuration and input/output variables.
        aggregated_results (dict): Dictionary of results from PC1 analysis, mapping outputs to set of input variables.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        list: A list of ModelDescription objects for each causal single-output neural network.
    """
    model_descriptions = list()

    for output, pc_alpha_dict in aggregated_results.items():
        print(output)
        if len(pc_alpha_dict) == 0:  # May be empty
            print("Empty results")
            pass
        for pc_alpha, pc_alpha_results in pc_alpha_dict.items():
            var_names = np.array(pc_alpha_results["var_names"])
            for threshold, parent_idxs in pc_alpha_results["parents"].items():
                parents = var_names[parent_idxs]
                model_description = ModelDescription(
                    output, parents, setup.nn_type, pc_alpha, threshold, setup=setup, seed=seed
                )
                model_descriptions.append(model_description)
    return model_descriptions


def generate_models(setup, threshold_dict=False, continue_training=False, seed=None):
    """Generates all neural network models specified in the setup.

    Args:
        setup (pcmasking.utils.setup.Setup): Setup object containing model configuration.
        threshold_dict (bool, optional): Threshold dictionary for input selection. Defaults to False.
        continue_training (bool, optional): Whether to continue training from previous checkpoints. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        list: A list of ModelDescription objects for each model.
    """
    model_descriptions = list()

    if setup.distribute_strategy == "mirrored":
        if not tf.config.get_visible_devices('GPU'):
            raise EnvironmentError(f"Cannot build and compile models with tf.distribute.MirroredStrategy "
                                   f"because Tensorflow found no GPUs.")
        print(f"\n\nBuilding and compiling models with tf.distribute.MirroredStrategy.", flush=True)
    else:
        print(f"\n\nBuilding and compiling models.", flush=True)

    training_pcmasking = setup.nn_type in ["PreMaskNet", "MaskNet"]
    if setup.do_single_nn or training_pcmasking:
        model_descriptions.extend(generate_all_single_nn(setup, continue_training, seed=seed))

    if setup.do_random_single_nn:
        collected_results, errors = aggregation.collect_results(setup, reuse=True)
        aggregation.print_errors(errors)

        aggregated_results, var_names_parents = aggregation.aggregate_results_for_numparents(
            collected_results, setup, thresholds_dict=setup.thrs_argv, random=setup.random
        )

        model_descriptions.extend(
            generate_all_causal_single_nn(setup, aggregated_results, seed=seed)
        )

    if setup.do_causal_single_nn:
        collected_results, errors = aggregation.collect_results(setup, reuse=True)
        aggregation.print_errors(errors)
        aggregated_results, var_names_parents = aggregation.aggregate_results(
            collected_results, setup, threshold_dict=threshold_dict)
        model_descriptions.extend(
            generate_all_causal_single_nn(setup, aggregated_results, seed=seed)
        )

    return model_descriptions


def generate_model_sherpa(setup, parents=False, pc_alpha=None, threshold=None, seed=None):
    """Generates neural network model for hyperparameter tuning via Sherpa"""

    if setup.do_causal_single_nn and parents == False:
        parents = get_parents_sherpa(setup)

    model_description = ModelDescription(
        setup.output, parents, setup.nn_type, pc_alpha, threshold, setup=setup, seed=seed
    )
    return model_description


def generate_input_list(setup):
    """Generates input list for hyperparameter tuning via Sherpa"""
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
    """Generates output list for hyperparameter tuning via Sherpa"""
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
