import argparse
import datetime
import time
from pathlib import Path

from pcmasking.neural_networks.models import generate_models
from pcmasking.neural_networks.training import train_all_models
from pcmasking.utils.main_utils import parse_str_to_bool_or_int, set_random_seed, load_fine_tune_weights
from pcmasking.utils.setup import SetupNeuralNetworks
from pcmasking.utils.tf_gpu_management import manage_gpu


def train_pcmasking(config_file, config_fine_tune, seed):
    argv = ["-c", config_file]
    setup = SetupNeuralNetworks(argv)

    model_descriptions = generate_models(setup, seed=seed)

    # If we are doing fine-tuning, we need to load the weights from trained PreMaskNet
    if setup.nn_type == "MaskNet" and config_fine_tune is not None:
        load_fine_tune_weights(config_fine_tune, model_descriptions, seed)

    train_all_models(model_descriptions, setup)


if __name__ == "__main__":
    """
    Main function to train PCMasking networks for all output variables.

    Command-line Arguments:
        -c, --config_file (str, required): Path to the YAML configuration file for neural network creation.
        -s, --seed (int or bool, optional): Integer value for random seed to ensure reproducibility. 
                                            Use 'False' to skip setting a random seed. Defaults to False.
        -f, --fine_tune_config (str, optional): Path to the configuration file for fine-tuning from a PreMaskNet model.
                                                If not provided, models are trained from scratch.

    Variables:
        yaml_config_file (Path): Path object representing the YAML configuration file.
        random_seed_parsed (int or bool): The parsed random seed value or False if not set.
        fine_tune_cfg (Path or None): Path to the fine-tuning configuration file, if provided.

    Raises:
        ArgumentError: If the provided configuration file is not a valid YAML file.

    Example:
        To train a PCMasking network with a specified configuration and seed:
        $ python main_train_pcmasking.py -c config.yml -s 42
        
        To train with fine-tuning:
        $ python main_train_pcmasking.py -c config.yml -s 42 -f fine_tune_cfg.yml

    Workflow:
        1. Parse command-line arguments and validate the configuration file.
        2. Manage GPU settings to allow memory growth or limit usage based on the system setup.
        3. Set a random seed for reproducibility if specified.
        4. Print the configuration details, including the YAML config file, fine-tuning config (if any), and random seed.
        5. Load the model setup from the configuration file.
        6. Generate models and optionally load fine-tuned weights.
        7. Train the model(s) based on the specified configurations.
    """
    parser = argparse.ArgumentParser(description="Train PCMasking networks for all output variables.")

    parser.add_argument("-c", "--config_file", help="YAML configuration file for neural network creation.",
                        required=True)
    parser.add_argument("-s", "--seed", help="Integer value for random seed. "
                                             "Use 'False' or leave out this option to not set a random seed.",
                        default=False, type=parse_str_to_bool_or_int, nargs='?', const=True)
    parser.add_argument("-f", "--fine_tune_config",
                        help="Config for trained PreMaskNet to load weights for fine-tuning.",
                        required=False, default=None, type=str, nargs='?')
    args = parser.parse_args()

    yaml_config_file = Path(args.config_file)
    random_seed_parsed = args.seed
    fine_tune_cfg = None if args.fine_tune_config == "" else args.fine_tune_config

    if not yaml_config_file.suffix == ".yml":
        parser.error(f"Configuration file must be YAML file (.yml). Got {yaml_config_file}")

    # GPU management: Allow memory growth if training is done on multiple GPUs, otherwise limit GPUs to single GPU
    manage_gpu(yaml_config_file)

    # Set random seed
    random_seed = set_random_seed(random_seed_parsed)

    print(f"\nYAML config file:    {yaml_config_file}")
    print(f"Fine-tuning config:    {fine_tune_cfg}")
    print(f"Random seed:           {random_seed}\n")

    print(f"\n\n{datetime.datetime.now()} --- Start PCMasking training.", flush=True)
    t_init = time.time()

    train_pcmasking(yaml_config_file, random_seed)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"{datetime.datetime.now()} Finished. Time: {t_total}")
