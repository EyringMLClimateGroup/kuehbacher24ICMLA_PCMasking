import tensorflow as tf

from neural_networks.models import ModelDescription


def generate_inputs_and_outputs(setup, inputs_file, outputs_file):
    inputs_list = _build_spcam_var_list(setup.spcam_inputs, setup.parents_idx_levs)
    outputs_list = _build_spcam_var_list(setup.spcam_outputs, setup.children_idx_levs)

    with open(inputs_file, 'w') as i_file:
        for var in inputs_list:
            i_file.write(f"{var}\n")
    print(f"Successfully wrote NN input variables to {inputs_file}.")

    with open(outputs_file, 'w') as o_file:
        for var in outputs_list:
            o_file.write(f"{var}\n")
    print(f"Successfully wrote NN output variables to {outputs_file}.")


def generate_single_nn_for_output_list(setup, inputs_list, outputs_list):
    model_descriptions = list()

    for output in outputs_list:
        model_description = ModelDescription(
            output, inputs_list, setup.nn_type, pc_alpha=None, threshold=None, setup=setup,
        )
        model_descriptions.append(model_description)
    return model_descriptions


def generate_models(setup, inputs, outputs):
    """ Generate all NN models specified in setup """
    model_descriptions = list()

    if setup.do_mirrored_strategy:
        if setup.do_mirrored_strategy and not tf.config.get_visible_devices('GPU'):
            raise EnvironmentError(f"Cannot build and compile models with tf.distribute.MirroredStrategy "
                                   f"because Tensorflow found no GPUs.")
        print(f"\n\nBuilding and compiling models with tf.distribute.MirroredStrategy.", flush=True)

    if setup.do_single_nn or setup.do_pca_nn or setup.do_castle_nn:
        model_descriptions.extend(generate_single_nn_for_output_list(setup, inputs, outputs))

    else:
        raise NotImplementedError("Splitting training over SLURM nodes only implemented for "
                                  "single NN, PCA NN, CASTLE NN.")
    return model_descriptions


def _build_spcam_var_list(spcam_vars, levels_idxs):
    var_list = list()

    for spcam_var in spcam_vars:
        if spcam_var.dimensions == 3:
            for level, _ in levels_idxs:
                # There's enough info to build a Variable_Lev_Metadata list
                # However, it could be better to do a bigger reorganization
                var_name = f"{spcam_var.name}-{round(level, 2)}"
                var_list.append(var_name)
        elif spcam_var.dimensions == 2:
            var_name = spcam_var.name
            var_list.append(var_name)
    return var_list
