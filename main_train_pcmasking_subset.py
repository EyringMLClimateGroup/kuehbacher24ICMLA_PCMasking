import argparse
import datetime
import time
from pathlib import Path

from pcmasking.neural_networks.models_split_over_nodes import generate_models
from pcmasking.neural_networks.training import train_all_models
from pcmasking.utils.main_utils import load_fine_tune_weights, save_masking_vector, save_history, read_txt_to_list, \
    parse_str_to_bool, parse_str_to_bool_or_int, set_random_seed
from pcmasking.utils.setup import SetupNeuralNetworks
from pcmasking.utils.tf_gpu_management import manage_gpu, set_gpu


def train_pcmasking_subset(config_file, nn_inputs_file, nn_outputs_file, train_indices, load_weights_from_ckpt,
                           continue_previous_training, config_fine_tune, seed):
    argv = ["-c", config_file]
    setup = SetupNeuralNetworks(argv)

    inputs = read_txt_to_list(nn_inputs_file)
    outputs = read_txt_to_list(nn_outputs_file)

    selected_outputs = [outputs[i] for i in train_indices]
    model_descriptions = generate_models(setup, inputs, selected_outputs, continue_training=continue_previous_training,
                                         seed=seed)

    # If we are doing fine-tuning, we need to load the weights from trained PreMaskNet
    if setup.nn_type == "MaskNet" and config_fine_tune is not None:
        load_fine_tune_weights(config_fine_tune, model_descriptions, seed, inputs=inputs, outputs=selected_outputs)

    history = train_all_models(model_descriptions, setup, from_checkpoint=load_weights_from_ckpt,
                               continue_training=continue_previous_training)

    # Save histories
    base_dir = Path(config_file).parent
    save_history(history, model_descriptions, selected_outputs, base_dir)

    # In case of PreMaskNet, save the masking vector for each variable
    if setup.nn_type == "PreMaskNet":
        save_masking_vector(model_descriptions, selected_outputs, base_dir)


if __name__ == "__main__":
    """
    Main function to train PCMasking networks for a subset of specified output variables.

    Command-line Arguments:
        -s, --seed (int, optional): Random seed for reproducibility. If not provided, a random seed is not set.
        -g, --gpu_index (int, optional): Index of the GPU to use for training. If not provided, all available GPUs are used.
        -f, --fine_tune_config (str, optional): YAML configuration file for fine-tuning from a trained PreMaskNet. If not provided, training will start from scratch.
        -c, --config_file (str, required): Path to the YAML configuration file for neural network creation.
        -i, --inputs_file (str, required): Path to the .txt file containing neural network input variables.
        -o, --outputs_file (str, required): Path to the .txt file containing neural network output variables.
        -x, --train_indices (str, required): Range of output variable indices in the format 'start-end' to specify which networks to train.
        -l, --load_ckpt (bool, required): Flag to load weights from a previous checkpoint during training.
        -t, --continue_training (bool, required): Flag to continue training from the previous session, resuming model and optimizer states.

    Variables:
        yaml_config_file (Path): Path object for the YAML configuration file.
        inputs_file (Path): Path object for the input variables .txt file.
        outputs_file (Path): Path object for the output variables .txt file.
        train_idx (list[int]): List of indices for the selected output variables to train.
        load_ckpt (bool): Whether to load model weights from a previous checkpoint.
        continue_training (bool): Whether to continue training from the previous session.
        random_seed_parsed (int or bool): Parsed random seed value.
        gpu_index (int or None): Index of the GPU to use for training, if provided.
        fine_tune_cfg (Path or None): Path to the fine-tuning YAML configuration file, if provided.

    Raises:
        ArgumentError: If the configuration file, inputs file, or outputs file have incorrect extensions.
        ValueError: If the range of train indices is incorrect.

    Example:
        $ python main_train_pcmasking_subset.py -c config.yml -i inputs.txt -o outputs.txt -x "1-10" -l False -t False -s 42 -g 0

    Workflow:
        1. Parse command-line arguments and validate file paths and extensions.
        2. Manage GPU settings based on the specified GPU index or multi-GPU configuration.
        3. Parse the range of output indices for training.
        4. Set a random seed for reproducibility if specified.
        5. Load the model setup and input/output variables.
        6. Generate models and optionally load fine-tuned weights.
        7. Train the model(s) based on the specified configurations.
        8. Save training history and masking vector if required.
    """
    parser = argparse.ArgumentParser(
        description="Trains PCMasking networks for only a subset of specified output variables.")
    parser.add_argument("-s", "--seed", help="Integer value for random seed. "
                                             "Use 'False' or leave out this option to not set a random seed.",
                        default=False, type=parse_str_to_bool_or_int, nargs='?', const=True)
    parser.add_argument("-g", "--gpu_index",
                        help="GPU index. If given, only the GPU specified by index will be used for training.",
                        required=False, default=False, type=int, nargs='?')
    parser.add_argument("-f", "--fine_tune_config",
                        help="Configuration file for previously trained PreMaskNet to load weights from for fine-tuning.",
                        required=False, default=None, type=str, nargs='?')

    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("-c", "--config_file", help="YAML configuration file for neural network creation.",
                               required=True)
    required_args.add_argument("-i", "--inputs_file", help=".txt file with NN inputs list.", required=True, type=str)
    required_args.add_argument("-o", "--outputs_file", help=".txt file with NN outputs list.", required=True, type=str)
    required_args.add_argument("-x", "--train_indices", help="Start and end index of outputs in outputs list, "
                                                             "specifying the neural networks that are to be trained. "
                                                             "Must be a string in the form 'start-end'.",
                               required=True, type=str)
    required_args.add_argument("-l", "--load_ckpt",
                               help="Boolean indicating whether to load weights from checkpoint from previous training.",
                               required=True, type=parse_str_to_bool)
    required_args.add_argument("-t", "--continue_training",
                               help="Boolean indicating whether to continue with previous training. The model "
                                    "(including optimizer) is loaded and the learning rate is initialized with the "
                                    "last learning rate from previous training.",
                               required=True, type=parse_str_to_bool)

    args = parser.parse_args()

    yaml_config_file = Path(args.config_file)
    inputs_file = Path(args.inputs_file)
    outputs_file = Path(args.outputs_file)
    train_idx = args.train_indices
    load_ckpt = args.load_ckpt
    continue_training = args.continue_training
    random_seed_parsed = args.seed
    gpu_index = args.gpu_index
    fine_tune_cfg = None if args.fine_tune_config == "" else args.fine_tune_config

    if not yaml_config_file.suffix == ".yml":
        parser.error(f"Configuration file must be YAML file (.yml). Got {yaml_config_file}")
    if not inputs_file.suffix == ".txt":
        parser.error(f"File with neural network inputs must be .txt file. Got {inputs_file}")
    if not outputs_file.suffix == ".txt":
        parser.error(f"File with neural network outputs must be .txt file. Got {outputs_file}")

    if fine_tune_cfg is not None:
        if not fine_tune_cfg.endswith(".yml"):
            parser.error(f"Fine-tuning configuration file must be YAML file (.yml). Got {fine_tune_cfg}")
        else:
            fine_tune_cfg = Path(fine_tune_cfg)

    # GPU management: Allow memory growth if training is done on multiple GPUs, otherwise limit GPUs to single GPU
    if gpu_index is None:
        manage_gpu(yaml_config_file)
    else:
        set_gpu(index=gpu_index)

    # Parse indices of outputs selected for training
    start, end = train_idx.split("-")
    train_idx = list(range(int(start), int(end) + 1))
    if not train_idx:
        raise ValueError("Given train indices were incorrect. Start indices must be smaller than end index. ")

    # Set random seed
    random_seed = set_random_seed(random_seed_parsed)

    print(f"\nYAML config file:      {yaml_config_file}")
    print(f"Input list .txt file:  {inputs_file}")
    print(f"Output list .txt file: {outputs_file}")
    print(f"Train indices:         {train_idx}")
    print(f"Fine-tuning config:    {fine_tune_cfg}")
    print(f"Random seed:           {random_seed}\n")

    print(f"\n\n{datetime.datetime.now()} --- Start training PCMasking networks.", flush=True)
    t_init = time.time()

    train_pcmasking_subset(yaml_config_file, inputs_file, outputs_file, train_idx, load_ckpt, continue_training,
                           fine_tune_cfg,
                           random_seed)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
