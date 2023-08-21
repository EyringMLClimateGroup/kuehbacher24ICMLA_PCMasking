import argparse
import datetime
import time
from pathlib import Path

import tensorflow as tf
from nni.experiment import Experiment

from main_train_castle_split_nodes import parse_str_to_bool, parse_str_to_bool_or_int, set_memory_growth_gpu


def tune_castle(config, inputs, outputs, indices, load_checkpoint, cont_training, seed, tuning_alg):
    search_space = {
        'num_hidden_layers': {'_type': 'choice', '_value': list(range(15))},
        'dense_units': {'_type': 'choice', '_value': [32, 64, 128, 256]},
        'activation_type': {'_type': 'choice', '_value': ['relu', 'tanh', 'leakyrelu']},
        'learning_rate': {'_type': 'choice', '_value': [0.001, 0.01, 0.1]},
        'learning_rate_schedule': {'_type': 'choice', '_value': [('exp', 5, 3), ('exp', 2, 1), ('plateau', 3)]},
        'lambda': {'_type': 'choice', '_value': [0.1, 1.0, 2.0]},
    }

    experiment = Experiment('local')
    experiment.config.trial_command = f"python -u main_train_castle_split_nodes_tuning.py -c {config} -i {inputs} -o {outputs} -x {indices} -l {load_checkpoint} -t {cont_training} -s {seed}"
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = search_space

    experiment.config.tuner.name = tuning_alg
    experiment.config.tuner.class_args['optimize_mode'] = 'minimize'

    experiment.config.max_trial_number = 10
    experiment.config.trial_concurrency = 10

    experiment.run(5848)


def parse_arguments():
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
    required_args.add_argument("-l", "--load_ckpt",
                               help="Boolean indicating whether to load weights from checkpoint from previous training.",
                               required=True, type=parse_str_to_bool)
    required_args.add_argument("-t", "--continue_training",
                               help="Boolean indicating whether to continue with previous training. The model "
                                    "(including optimizer) is loaded and the learning rate is initialized with the "
                                    "last learning rate from previous training.",
                               required=True, type=parse_str_to_bool)

    required_args.add_argument("-u", "--tuner", help="Tuning algorithm to be used (e.g. TPE, Random, Hyperband).",
                               required=True, type=str)
    args = parser.parse_args()
    yaml_config_file = Path(args.config_file)
    inputs_file = Path(args.inputs_file)
    outputs_file = Path(args.outputs_file)
    train_idx = args.train_indices
    load_ckpt = args.load_ckpt
    continue_training = args.continue_training
    random_seed_parsed = args.seed
    tuning_alg = args.tuner

    return yaml_config_file, inputs_file, outputs_file, train_idx, load_ckpt, continue_training, random_seed_parsed, tuning_alg


if __name__ == "__main__":
    # Allow memory growth for GPUs (this seems to be very important, because errors occur otherwise)
    if len(tf.config.list_physical_devices("GPU")):
        print(f"\nAllow memory growth on GPUs.", flush=True)
        set_memory_growth_gpu()

    yaml_config_file, inputs_file, outputs_file, train_idx, load_ckpt, continue_training, random_seed, tuner = parse_arguments()

    print(f"\n\n{datetime.datetime.now()} --- Start CASTLE tuning.", flush=True)
    t_init = time.time()

    tune_castle(yaml_config_file, inputs_file, outputs_file, train_idx, load_ckpt, continue_training, tuner)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
