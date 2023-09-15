import argparse
import datetime
import time
from pathlib import Path
import yaml

import tensorflow as tf
from nni.experiment import Experiment

from main_train_castle_split_nodes import parse_str_to_bool_or_int, set_memory_growth_gpu


# After experiment is done, run nni.experiment.Experiment.view(experiment_id, port=32325) to restart web portal

def tune_castle(config, inputs, outputs, indices, seed, tuning_alg, tuning_metric, search_space):
    experiment = Experiment('local')
    experiment.config.trial_command = f"python -u main_train_castle_split_nodes_tuning.py -c {config} -i {inputs} -o {outputs} -x {indices} -p {tuning_metric} -s {seed}"
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = search_space

    experiment.config.tuner.name = tuning_alg
    experiment.config.tuner.class_args['optimize_mode'] = 'minimize'

    experiment.config.max_trial_number = 40
    experiment.config.max_experiment_duration = "718m"  # less than 13h, so that the experiment finishes before the job limit
    experiment.config.trial_concurrency = 40
    experiment.config.trial_gpu_number = 4

    # Set to false if multiple exp
    experiment.config.training_service.use_active_gpu = True

    experiment.run(port=32325)


def read_yaml(yaml_file):
    with open(yaml_file, "r") as read_stream:
        search_space_config = yaml.safe_load(read_stream)
    return search_space_config["search_space"]


def parse_arguments():
    """
    Parses command line arguments.

    Returns:
        Parsed command line arguments:
        - YAML network configuration file (str)
        - Network inputs .txt file (str)
        - Network outputs .txt file (str)
        - Indices for networks to be trained (str)
        - Random seed (int/bool)
        - Tuning algorithm (str)
        - Tuning metric (str)
        - Tuning search space (dict)

    """
    parser = argparse.ArgumentParser(description="Generates .txt files for neural network input and output "
                                                 "variables for specific setup configuration.")
    parser.add_argument("-s", "--seed", help="Integer value for random seed. "
                                             "Use 'False' or leave out this option to not set a random seed.",
                        default=False, type=parse_str_to_bool_or_int, nargs='?', const=True)

    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("-c", "--config_file", help="YAML configuration file for neural network creation.",
                               required=True)
    required_args.add_argument("-i", "--inputs_file", help=".txt file with NN inputs list.", required=True, type=str)
    required_args.add_argument("-o", "--outputs_file", help=".txt file with NN outputs list.", required=True, type=str)
    required_args.add_argument("-x", "--train_indices", help="Start and end index of outputs in outputs list, "
                                                             "specifying the neural networks that are to be trained. "
                                                             "Must be a string in the form 'start-end'.",
                               required=True, type=str)

    required_args.add_argument("-u", "--tuner", help="Tuning algorithm to be used (e.g. TPE, Random, Hyperband).",
                               required=True, type=str)
    required_args.add_argument("-p", "--tuning_metric",
                               help="Metric used to measure tuning performance (e.g. 'val_loss', 'val_prediction_loss').",
                               required=True, type=str)
    required_args.add_argument("-e", "--search_space", help="YAML file with tuning search space.",
                               required=True, type=read_yaml)

    args = parser.parse_args()

    yaml_config_file = Path(args.config_file)
    inputs_file = Path(args.inputs_file)
    outputs_file = Path(args.outputs_file)
    train_idx = args.train_indices
    random_seed_parsed = args.seed
    tuning_alg = args.tuner
    tuning_metric = args.tuning_metric
    search_space = args.search_space

    return yaml_config_file, inputs_file, outputs_file, train_idx, random_seed_parsed, tuning_alg, tuning_metric, search_space


if __name__ == "__main__":
    # Allow memory growth for GPUs (this seems to be very important, because errors occur otherwise)
    if len(tf.config.list_physical_devices("GPU")):
        print(f"\nAllow memory growth on GPUs.", flush=True)
        set_memory_growth_gpu()

    cfg_file, inputs_file, outputs_file, train_idx, random_seed, tuner, metric, tuning_search_space = parse_arguments()

    print(f"\n\n{datetime.datetime.now()} --- Start CASTLE tuning.", flush=True)
    t_init = time.time()

    tune_castle(cfg_file, inputs_file, outputs_file, train_idx, random_seed, tuner, metric, tuning_search_space)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
