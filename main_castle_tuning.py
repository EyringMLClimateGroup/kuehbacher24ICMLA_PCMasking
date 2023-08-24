import argparse
import datetime
import time
from pathlib import Path
import socket

import tensorflow as tf
from nni.experiment import Experiment

from main_train_castle_split_nodes import parse_str_to_bool_or_int, set_memory_growth_gpu


def tune_castle(config, inputs, outputs, indices, seed, tuning_alg, tuning_metric):
    search_space = {
        'num_hidden_layers': {'_type': 'choice', '_value': list(range(15))},
        'dense_units': {'_type': 'choice', '_value': [32, 64, 128, 256]},
        'activation_type': {'_type': 'choice', '_value': ['relu', 'tanh', 'leakyrelu']},
        'learning_rate': {'_type': 'choice', '_value': [0.001, 0.01, 0.1]},
        'learning_rate_schedule': {'_type': 'choice',
                                   '_value': [('exp', 5, 3), ('exp', 2, 1), ('plateau', 0.1), ('plateau', 0.5)]},
        'lambda': {'_type': 'choice', '_value': [0.1, 1.0, 2.0]},
    }

    experiment = Experiment('local')
    experiment.config.trial_command = f"python -u main_train_castle_split_nodes_tuning.py -c {config} -i {inputs} -o {outputs} -x {indices} -p {tuning_metric} -s {seed}"
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = search_space

    experiment.config.tuner.name = tuning_alg
    experiment.config.tuner.class_args['optimize_mode'] = 'minimize'

    experiment.config.max_trial_number = 30
    experiment.config.max_experiment_duration = "12h"
    experiment.config.trial_concurrency = 10
    experiment.config.trial_gpu_number = 4

    # Set to false if multiple exp
    experiment.config.training_service.use_active_gpu = True

    # Find open port
    port = get_open_port(ip="127.0.0.1", port=8080)

    experiment.run(port=port)


def get_open_port(ip, port, tries=10):
    """
    Returns an open network port at `ip`.
    If the given `port` is not open, it iteratively increases the port number
    by 1 for `tries` times to find an open port.

    Args:
        ip (str): IP address.
        port (int): Port number.
        tries (int): Number of times the port number is increased by if the current
          port is not open. Default: 10.

    Returns:
        Port number for open network port (int).

    Raises:
        ConnectionError: If not open network port could be found.

    """
    for _ in range(tries):
        if _port_is_open(ip, port):
            print(f"\nUsing open port {port} at ip {ip}.\n")
            return port
        else:
            port = port + 1
    raise ConnectionError("Could not find open port.")


def _port_is_open(ip, port):
    timeout = 30

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((ip, port))
        s.shutdown(socket.SHUT_RDWR)
        return True
    except:
        return False
    finally:
        s.close()


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

    args = parser.parse_args()

    yaml_config_file = Path(args.config_file)
    inputs_file = Path(args.inputs_file)
    outputs_file = Path(args.outputs_file)
    train_idx = args.train_indices
    random_seed_parsed = args.seed
    tuning_alg = args.tuner
    tuning_metric = args.tuning_metric

    return yaml_config_file, inputs_file, outputs_file, train_idx, random_seed_parsed, tuning_alg, tuning_metric


if __name__ == "__main__":
    # Allow memory growth for GPUs (this seems to be very important, because errors occur otherwise)
    if len(tf.config.list_physical_devices("GPU")):
        print(f"\nAllow memory growth on GPUs.", flush=True)
        set_memory_growth_gpu()

    cfg_file, inputs_file, outputs_file, train_idx, random_seed, tuner, metric = parse_arguments()

    print(f"\n\n{datetime.datetime.now()} --- Start CASTLE tuning.", flush=True)
    t_init = time.time()

    tune_castle(cfg_file, inputs_file, outputs_file, train_idx, random_seed, tuner, metric)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
