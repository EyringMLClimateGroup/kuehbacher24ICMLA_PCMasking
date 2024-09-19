import argparse
import datetime
import gc
import time
from pathlib import Path

import numpy as np

from pcmasking.neural_networks.models_split_over_nodes import generate_models
from pcmasking.neural_networks.training import train_all_models
from pcmasking.utils.main_utils import read_txt_to_list, load_fine_tune_weights, parse_str_to_bool, \
    parse_str_to_bool_or_int, save_threshold_histories, get_min_threshold_last_metric_element, \
    get_min_threshold_min_metric_element
from pcmasking.utils.setup import SetupNeuralNetworks
from pcmasking.utils.tf_gpu_management import manage_gpu, set_tf_random_seed, set_gpu


def train_mask_net_thresholds(config_file, nn_inputs_file, nn_outputs_file, train_index, percentile,
                              load_weights_from_ckpt, continue_previous_training, config_fine_tune, seed):
    base_dir = Path(config_file).parent

    argv = ["-c", config_file]
    setup = SetupNeuralNetworks(argv)
    if setup.nn_type is not "MaskNet":
        raise ValueError(f"Training with thresholds is only valid for MaskNet. Got network type {setup.nn_type}")

    inputs = read_txt_to_list(nn_inputs_file)
    outputs = read_txt_to_list(nn_outputs_file)

    selected_output = outputs[train_index]

    # Load masking vector and compute thresholds
    masking_vector_file = Path(str(setup.masking_vector_file).format(var=selected_output))
    print(f"\nLoading masking vector {(Path(*Path(masking_vector_file).parts[-4:]))}\n to get threshold bounds.")
    masking_vector = np.load(masking_vector_file)

    lower_bound = 1e-4
    upper_bound = np.percentile(masking_vector, percentile)
    print(f"\nUsing thresholds between {lower_bound} and {upper_bound}.")

    thresholds = np.linspace(start=lower_bound, stop=upper_bound, num=20, endpoint=False)

    # Run training for each threshold
    history_per_threshold = dict()
    min_threshold = lower_bound

    for idx, t in enumerate(thresholds):
        t = round(t, 4)

        if t == 0.:
            print(f"\n\n--- Using minimum threshold t={min_threshold}.\n\n")
            t = min_threshold

        if t in history_per_threshold.keys():
            print(f"\n\n\n--- Skipping threshold number {idx + 1} because value already occurred (t={t}).\n\n")
            continue

        print(f"\n\n\n--- Training with threshold number {idx + 1} with value {t}.\n\n")

        setup.mask_threshold = t

        model_descriptions = generate_models(setup, inputs, [selected_output],
                                             continue_training=continue_previous_training,
                                             seed=seed)

        # If we are doing fine-tuning, we need to load the weights from trained PreMaskNet
        if config_fine_tune is not None:
            load_fine_tune_weights(config_fine_tune, model_descriptions, seed, inputs=inputs, outputs=selected_output)

        h = train_all_models(model_descriptions, setup, from_checkpoint=load_weights_from_ckpt,
                             continue_training=continue_previous_training)
        history_per_threshold[t] = h[selected_output].history

        gc.collect()

    print(f"\n\n\n--- Finished training for all thresholds.")
    print("\nThreshold = {} for best last training loss = {}".format(
        *get_min_threshold_last_metric_element(history_per_threshold, "loss")))
    print("\nThreshold = {} for best validation loss = {}".format(
        *get_min_threshold_min_metric_element(history_per_threshold, "val_loss")))
    print("\n---")

    # Save history
    save_threshold_histories(base_dir, history_per_threshold, model_descriptions)


if __name__ == "__main__":
    """
    Main function to train MaskNet models for a single output variable for multiple thresholds.

    Command-line Arguments:
        -s, --seed (int, optional): Random seed for reproducibility. Defaults to 'False' or 42 if set.
        -g, --gpu_index (int, optional): Index of the GPU to be used for training. Defaults to None for multi-GPU training.
        -f, --fine_tune_config (str, optional): YAML configuration file for loading fine-tuned weights from PreMaskNet.
        -c, --config_file (str, required): YAML configuration file for neural network creation.
        -i, --inputs_file (str, required): Path to the .txt file containing neural network input variables.
        -o, --outputs_file (str, required): Path to the .txt file containing neural network output variables.
        -x, --train_index (int, required): Index of the network output variable to train in the output list.
        -r, --percentile (int, required): Percentile (between 50-100) used as the upper bound for threshold selection.
        -l, --load_ckpt (bool, required): Boolean to load weights from a previous checkpoint.
        -t, --continue_training (bool, required): Boolean to continue from a previous training session.

    Variables:
        yaml_config_file (Path): Path object for the YAML configuration file.
        inputs_file (Path): Path object for the input variables .txt file.
        outputs_file (Path): Path object for the output variables .txt file.
        train_idx (int): Index of the selected output variable in the outputs list for training.
        percentile (int): Percentile value used for determining the upper threshold bound.
        load_ckpt (bool): Whether to load model weights from a previous checkpoint.
        continue_training (bool): Whether to continue from the last training session.
        random_seed_parsed (bool or int): Parsed random seed value.
        gpu_index (int or None): GPU index to use for training, if provided.
        fine_tune_cfg (Path or None): YAML file path for fine-tuning from a PreMaskNet model.

    Raises:
        ArgumentError: Raised if input/output files or configuration files have incorrect extensions.
        ValueError: Raised if the percentile is not an integer between 50-100.

    Example:
        $ python main_train_mask_net_thresholds.py -c config.yml -i inputs.txt -o outputs.txt -x 0 -r 75 -l False -t False -f fine_tune_cfg.yml -s 42

    Workflow:
        1. Parse command-line arguments and validate file paths and extensions.
        2. Manage GPU settings based on the specified GPU index or multi-GPU configuration.
        3. Set a random seed for reproducibility if provided.
        4. Load the MaskNet setup and input/output variables.
        5. Determine threshold bounds.
        6. Optionally load fine-tuned weights from PreMaskNet if specified.
        7. Train the model, save training history, and print best threshold metrics.
    """
    parser = argparse.ArgumentParser(
        description="Trains MaskNet models for a single output variable for multiple thresholds.")
    parser.add_argument("-s", "--seed", help="Integer value for random seed. "
                                             "Use 'False' or leave out this option to not set a random seed.",
                        default=False, type=parse_str_to_bool_or_int, nargs='?', const=True)
    parser.add_argument("-g", "--gpu_index",
                        help="GPU index. If given, only the GPU specified by index will be used for training.",
                        required=False, default=False, type=int, nargs='?')
    parser.add_argument("-f", "--fine_tune_config",
                        help="Config for trained PreMaskNet to load weights for fine-tuning.",
                        required=False, default=None, type=str, nargs='?')

    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("-c", "--config_file", help="YAML configuration file for neural network creation.",
                               required=True)
    required_args.add_argument("-i", "--inputs_file", help=".txt file with NN inputs list.", required=True, type=str)
    required_args.add_argument("-o", "--outputs_file", help=".txt file with NN outputs list.", required=True, type=str)
    required_args.add_argument("-x", "--train_index", help="Index of network output variable in outputs list (int).",
                               required=True, type=int)
    required_args.add_argument("-r", "--percentile", help="Percentile of masking vector values used as upper bounds "
                                                          "for thresholds. Integer between 50-100.",
                               required=True, type=int)
    required_args.add_argument("-l", "--load_ckpt",
                               help="Boolean indicating whether to load weights from checkpoint from previous training.",
                               required=True, type=parse_str_to_bool)
    required_args.add_argument("-t", "--continue_training",
                               help="Boolean indicating whether to continue with previous training. The model "
                                    "(including optimizer) is loaded and the learning rate is initialized with the "
                                    "last learning rate from previous training.",
                               required=True, type=parse_str_to_bool)

    args = parser.parse_args()

    # Parse arguments
    yaml_config_file = Path(args.config_file)
    inputs_file = Path(args.inputs_file)
    outputs_file = Path(args.outputs_file)
    train_idx = args.train_index
    percentile = args.percentile
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

    # Check percentile value
    if not isinstance(percentile, int) or (not 50 <= percentile <= 100):
        raise ValueError("Given percentile was incorrect. Must be an integer value between 50-100.")

    # Set random seed
    if random_seed_parsed is False:
        random_seed = None
    else:
        if random_seed_parsed is True:
            random_seed = 42
        else:
            random_seed = random_seed_parsed
        set_tf_random_seed(random_seed)

    print(f"\nYAML config file:      {yaml_config_file}")
    print(f"Input list .txt file:  {inputs_file}")
    print(f"Output list .txt file: {outputs_file}")
    print(f"Train index:           {train_idx}")
    print(f"Percentile:            {percentile}")
    print(f"Fine-tuning config:    {fine_tune_cfg}")
    print(f"Random seed:           {random_seed}\n")

    print(f"\n\n{datetime.datetime.now()} --- Start model training.", flush=True)
    t_init = time.time()

    train_mask_net_thresholds(yaml_config_file, inputs_file, outputs_file, train_idx, percentile, load_ckpt,
                              continue_training, fine_tune_cfg, random_seed)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
