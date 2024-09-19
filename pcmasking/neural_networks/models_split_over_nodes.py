"""
These functions are used to generate ModelDescriptions objects when training is done over multiple HPC nodes
"""
import tensorflow as tf

from pcmasking.neural_networks.models import ModelDescription
from pcmasking.utils.variable import Variable_Lev_Metadata


def generate_single_nn_for_output_list(setup, inputs_list, outputs_list, continue_training, seed=None):
    """
    Generates ModelDescription object with a list of single-output neural networks for the specified output variables.

    Args:
        setup (pcmasking.utils.setup.Setup): Setup object containing the model configuration.
        inputs_list (list): List of input variables.
        outputs_list (list): List of output variables.
        continue_training (bool): Whether to continue training from a previous checkpoint.
        seed (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        list: A list of ModelDescription objects representing the generated models.
    """
    model_descriptions = list()

    for output in outputs_list:
        model_description = ModelDescription(
            output, inputs_list, setup.nn_type, pc_alpha=None, threshold=None, setup=setup,
            continue_training=continue_training, seed=seed
        )
        model_descriptions.append(model_description)
    return model_descriptions


def generate_models(setup, inputs, outputs, continue_training=False, seed=None):
    """Generates all neural network models as specified in the setup.

    Args:
        setup (pcmasking.utils.setup.Setup): Setup object containing the model configuration.
        inputs (list): List of input variables.
        outputs (list): List of output variables.
        continue_training (bool, optional): Whether to continue training from a previous checkpoint. Defaults to False.
        seed (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        list: A list of ModelDescription objects representing the generated models.

    Raises:
        NotImplementedError: If training across SLURM nodes is required but not implemented for the model type.
        EnvironmentError: If no GPUs are available and MirroredStrategy is used.
    """
    model_descriptions = list()

    if setup.distribute_strategy == "mirrored":
        if not tf.config.get_visible_devices('GPU'):
            raise EnvironmentError(f"Cannot build and compile models with tf.distribute.MirroredStrategy "
                                   f"because Tensorflow found no GPUs.")
        print(f"\n\nBuilding and compiling models with tf.distribute.MirroredStrategy.", flush=True)

    generating_pcmasking = setup.nn_type in ["PreMaskNet", "MaskNet"]

    if setup.do_single_nn or generating_pcmasking:
        model_descriptions.extend(
            generate_single_nn_for_output_list(setup, inputs, outputs, continue_training, seed=seed))

    else:
        raise NotImplementedError("Splitting training over SLURM nodes only implemented for single NN, "
                                  "PreMaskNet, and MaskNet.")
    return model_descriptions


def write_inputs_and_outputs_lists(setup, inputs_file, outputs_file):
    """Writes the input and output variable lists to the specified text files.

    Args:
        setup (pcmasking.utils.setup.Setup): Setup object containing the model configuration.
        inputs_file (str): Path to the text file where input variables should be written.
        outputs_file (str): Path to the text file where output variables should be written.

    Returns:
        None
    """
    inputs_list = _build_spcam_var_list('in', setup)
    outputs_list = _build_spcam_var_list('out', setup)

    with open(inputs_file, 'w') as i_file:
        for var in inputs_list:
            i_file.write(f"{var}\n")
    print(f"Successfully wrote NN input variables to {inputs_file}.\n")

    with open(outputs_file, 'w') as o_file:
        for var in outputs_list:
            o_file.write(f"{var}\n")
    print(f"Successfully wrote NN output variables to {outputs_file}.\n")


def write_outputs_mapping(setup, txt_file):
    """Writes mapping between output variable name and output variable save file name to a text file.

    Args:
        setup (pcmasking.utils.setup.Setup): Setup object containing the output configuration.
        txt_file (str): Path to the text file where the output mapping should be written.

    Returns:
        None
    """
    output_var_list = _build_spcam_var_list('out', setup)

    def _get_filename(output_var):
        """ Generate a filename to save the model """
        i_var = setup.output_order.index(output_var.var)
        i_level = output_var.level_idx
        if i_level is None:
            i_level = 0
        return f"{i_var}_{i_level}"

    output_vars_dict = {_get_filename(variable): str(variable) for variable in output_var_list}

    with open(txt_file, 'w') as file:
        for key, value in output_vars_dict.items():
            file.write(f"{key}: {value}\n")
    print(f"Successfully wrote output variables mapping to {txt_file}.\n")


def _build_spcam_var_list(var_type, setup):
    """Builds a list of SPCAM variables based on their type (input or output).

    Args:
        var_type (str): Type of variable to build ('in' for inputs or 'out' for outputs).
        setup (pcmasking.utils.setup.Setup): Setup object containing the input/output variable configuration.

    Returns:
        list: A list of Variable_Lev_Metadata objects representing the variables.

    Raises:
        ValueError: If the provided var_type is not 'in' or 'out'.
    """
    if var_type == 'in':
        spcam_vars = setup.spcam_inputs
        levels_idxs = setup.parents_idx_levs
        order_list = setup.input_order_list
    elif var_type == 'out':
        spcam_vars = setup.spcam_outputs
        levels_idxs = setup.children_idx_levs
        order_list = None
    else:
        raise ValueError(f"Unknown variable type {var_type}. Must be one of ['in', 'out']")

    var_list = list()

    for spcam_var in spcam_vars:
        if spcam_var.dimensions == 3:
            for level, _ in levels_idxs:
                var_name = f"{spcam_var.name}-{round(level, 2)}"
                var_list.append(var_name)
        elif spcam_var.dimensions == 2:
            var_name = spcam_var.name
            var_list.append(var_name)

    if var_type == 'in':
        var_list = sorted([Variable_Lev_Metadata.parse_var_name(p) for p in var_list],
                          key=lambda x: order_list.index(x))
    else:
        var_list = [Variable_Lev_Metadata.parse_var_name(p) for p in var_list]

    return var_list
