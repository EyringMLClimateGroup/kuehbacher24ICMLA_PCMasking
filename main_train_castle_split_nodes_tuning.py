# noinspection PyUnresolvedReferences
from utils.tf_gpu_management import set_memory_growth_gpu, limit_single_gpu

import argparse
import datetime
import os
import re
import time
from pathlib import Path

import tensorflow as tf
import yaml
import ray
from ray import train
from ray import tune
from ray.tune.search import bayesopt
from ray.tune.search import hebo

from neural_networks.models_split_over_nodes import generate_models
from neural_networks.training import train_all_models
from utils.setup import SetupNeuralNetworks


def objective_wrapper(nn_config_file, nn_inputs_file, nn_outputs_file, var_idx, seed, metric):
    argv = ["-c", nn_config_file]
    setup = SetupNeuralNetworks(argv)

    # Set setup output path with seed
    setup.nn_output_path = Path(str(setup.nn_output_path) + f"_s{seed}")
    setup.tensorboard_folder = Path(str(setup.tensorboard_folder) + f"_s{seed}")

    inputs = _read_txt_to_list(nn_inputs_file)
    outputs = _read_txt_to_list(nn_outputs_file)

    selected_output = [outputs[var_idx]]

    print(f"\n\nSelected output: {selected_output}")

    def objective(config):
        print(f"\n\nCurrent config: \n{config}")

        # Set setup arguments for tuning
        setup.hidden_layers = parse_hidden_layers(config["hidden_layers"])
        setup.activation = config["activation"]
        setup.init_lr = config["learning_rate"]
        setup.lr_schedule = parse_lr_schedule(config["learning_rate_schedule"])
        setup.lambda_sparsity = float(config["lambda_sparsity"])
        if config["kernel_initializer"] == "RandomNormal":
            setup.kernel_initializer_input_layers = {"initializer": "RandomNormal",
                                                     "mean": 0.0, "std": 0.01}
            setup.kernel_initializer_hidden_layers = {"initializer": "RandomNormal",
                                                      "mean": 0.0, "std": 0.1}
            setup.kernel_initializer_output_layers = {"initializer": "RandomNormal",
                                                      "mean": 0.0, "std": 0.01}
        elif config["kernel_initializer"] == "GlorotUniform":
            setup.kernel_initializer_input_layers = {"initializer": "GlorotUniform"}
            setup.kernel_initializer_hidden_layers = {"initializer": "GlorotUniform"}
            setup.kernel_initializer_output_layers = {"initializer": "GlorotUniform"}
        elif config["kernel_initializer"] == "MixRandomNormalGlorot":
            setup.kernel_initializer_input_layers = {"initializer": "RandomNormal",
                                                     "mean": 0.0, "std": 0.01}
            setup.kernel_initializer_hidden_layers = {"initializer": "GlorotUniform"}
            setup.kernel_initializer_output_layers = {"initializer": "GlorotUniform"}
        else:
            raise ValueError(f"Unknown value for kernel initializer: {config['kernel_initializer']}. "
                             f"Configured only for RandomNormal, GlorotUniform and MixRandomNormalGlorot.")

        # Create model and train it
        model_descriptions = generate_models(setup, inputs, selected_output, seed=seed)

        histories = train_all_models(model_descriptions, setup, from_checkpoint=False, continue_training=False,
                                     save_learning_rate=False)

        # Report the metric result
        final_metric = histories[selected_output[0]].history[metric][-1]

        return {metric: final_metric}

    return objective


def tune_model(nn_config_file, nn_inputs_file, nn_outputs_file, var_idx, search_alg, metric, search_space, seed,
               experiment_name="tuning_results", storage_path="~/ray_results", restore_dir=""):
    objective_with_resources = tune.with_resources(
        objective_wrapper(nn_config_file, nn_inputs_file, nn_outputs_file, var_idx, seed, metric), {"gpu": 1})

    asha_scheduler = tune.schedulers.ASHAScheduler(
        time_attr='training_iteration',
        metric=metric,
        mode='min',
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1,
    )

    if restore_dir != "":
        print(f"\n\nRestoring from directory {restore_dir}")
        search_alg.restore_from_dir(restore_dir)

        tuner = tune.Tuner(
            objective_with_resources,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                scheduler=asha_scheduler,
                num_samples=-1,
                time_budget_s=datetime.timedelta(hours=3.8)  # compute limits jsc/dkrz: 24h/12h
            ),
            run_config=train.RunConfig(
                name=f"{experiment_name}_s{seed}_restore_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                storage_path=storage_path,
            ),
        )
    else:
        tuner = tune.Tuner(
            objective_with_resources,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                scheduler=asha_scheduler,
                num_samples=-1,
                time_budget_s=datetime.timedelta(hours=11.8)  # compute limits jsc/dkrz: 24h/12h
            ),
            run_config=train.RunConfig(
                name=f"{experiment_name}_s{seed}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                storage_path=storage_path,
            ),
            param_space=search_space,
        )
    results = tuner.fit()
    print(f"\n\nFinished tuning. Results: \n{results}\n")


def get_tuning_search_alg(tuner, metric, seed):
    # If you select a tuner, also check that the tuning scheduler is compatible.
    # Currently tune.schedulers.ASHAScheduler is used
    if tuner == "BasicVariantGenerator":
        search_alg = tune.search.basic_variant.BasicVariantGenerator(random_state=seed)
    elif tuner == "BayesOptSearch":
        search_alg = bayesopt.BayesOptSearch(metric=metric, mode="min", random_state=seed)
    elif tuner == "HEBO":
        search_alg = hebo.HEBOSearch(metric=metric, mode="min", random_state_seed=seed)
    else:
        raise NotImplementedError(f"Tuning algorithm {tuner} not implemented.")
    return search_alg


def parse_search_space(yaml_file):
    # Read yaml file
    with open(yaml_file, "r") as read_stream:
        search_space_config = yaml.safe_load(read_stream)
    search_space = search_space_config["search_space"]
    # Convert learning rates to float, because yaml doesn't recognize 1e-3 as float
    search_space["learning_rate"]["_value"] = [float(lr) for lr in search_space["learning_rate"]["_value"]]
    search_space["lambda_sparsity"]["_value"] = [float(ls) for ls in search_space["lambda_sparsity"]["_value"]]

    # for schedule in search_space["learning_rate_schedule"]["_value"]:
    #     if schedule["schedule"] == "linear":
    #         schedule["end_lr"] = float(schedule["end_lr"])

    ray_search_space = dict()

    # Parse search space config to ray tune search space
    # I'm currently only using choice sampling.
    # Check https://docs.ray.io/en/latest/tune/api/search_space.html#tune-search-space
    # for other options
    for key, value in search_space.items():
        if value["_type"] == "choice":
            ray_search_space[key] = tune.choice(value["_value"])
        else:
            raise NotImplementedError(f"Search space sampling {value['_type']} for parameter {key} is not implemented.")

    return ray_search_space


def parse_lr_schedule(lr_schedule_str):
    lr_schedule = dict()
    key_values = [s.strip(" ") for s in lr_schedule_str.split(",")]
    for kv in key_values:
        key, value = [s.strip(" ") for s in kv.split(":")]
        lr_schedule[key] = value

    if lr_schedule["schedule"] == "exponential":
        lr_schedule["step"] = int(lr_schedule["step"])
        lr_schedule["divide"] = int(lr_schedule["divide"])
    elif lr_schedule["schedule"] == "linear":
        lr_schedule["decay_steps"] = int(lr_schedule["decay_steps"])
        lr_schedule["end_lr"] = float(lr_schedule["end_lr"])
    else:
        raise NotImplementedError(f"Parsing not implemented for learning rate schedule {lr_schedule['schedule']}")
    return lr_schedule


def parse_hidden_layers(hidden_layers_str):
    return [int(s.strip(" ")) for s in re.split("\[|,|\]", hidden_layers_str)[1:-1]]


def _read_txt_to_list(txt_file):
    line_list = list()
    with open(txt_file, 'r') as f:
        for line in f:
            line_list.append(line.rstrip())
    return line_list


def parse_random_seed(seed):
    if seed.lower() in ('yes', 'true', 't', 'y', '1'):
        seed = True
    elif seed.lower() in ('no', 'false', 'f', 'n', '0'):
        seed = False

    if isinstance(seed, bool):
        if seed:
            return 42
        else:
            return None
    else:
        try:
            return int(seed)
        except ValueError:
            raise ValueError(f"Could not parse random seed {seed}. See option -h for help.")


def read_nn_type(nn_config):
    with open(nn_config, "r") as read_stream:
        cfg = yaml.safe_load(read_stream)
        nn_type = cfg['nn_type']
    del cfg

    return nn_type


if __name__ == "__main__":
    # Allow memory growth for GPUs (this seems to be very important, because errors occur otherwise)
    # set_memory_growth_gpu()

    # Initialize ray tune dashboard host to make dashboard accessible via ssh tunneling
    ray.init(dashboard_host='0.0.0.0')

    parser = argparse.ArgumentParser(description="Tunes CASTLE models.")
    parser.add_argument("-s", "--seed", help="Integer value for random seed. "
                                             "Use 'False' or leave out this option to not set a random seed.",
                        default=False, type=parse_random_seed, nargs='?', const=True)
    parser.add_argument("-d", "--restore_directory", help="Experiment directory when a previous experiment is to be "
                                                          "restored and continued (str).",
                        default="", type=str, nargs='?', const=True)

    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("-c", "--config_file", help="YAML configuration file for neural network creation.",
                               required=True)
    required_args.add_argument("-i", "--inputs_file", help=".txt file with NN inputs list.", required=True, type=str)
    required_args.add_argument("-o", "--outputs_file", help=".txt file with NN outputs list.", required=True, type=str)
    required_args.add_argument("-x", "--var_index",
                               help="Index of the output variable in outputs_file.txt for which to run tuning (int).",
                               required=True, type=int)
    required_args.add_argument("-u", "--tuner",
                               help="Tuning algorithm to be used (e.g. BasicVariantGenerator, BayesOptSearch, HEBO)",
                               required=True, type=str)
    required_args.add_argument("-p", "--tuning_metric",
                               help="Metric used to measure tuning performance (e.g. 'val_loss', 'val_prediction_loss').",
                               required=True, type=str)
    required_args.add_argument("-e", "--search_space", help="YAML file with tuning search space.",
                               required=True, type=parse_search_space)

    args = parser.parse_args()

    yaml_config_file = Path(args.config_file)
    inputs_file = Path(args.inputs_file)
    outputs_file = Path(args.outputs_file)
    var_index = args.var_index
    random_seed = args.seed
    restore_directory = args.restore_directory

    print(f"\n\nRandom seed is: seed={random_seed}", flush=True)
    tf.random.set_seed(random_seed)

    tuning_metric = args.tuning_metric
    search_space = args.search_space
    tuning_search_alg = get_tuning_search_alg(args.tuner, tuning_metric, random_seed)

    if not yaml_config_file.suffix == ".yml":
        parser.error(f"Configuration file must be YAML file (.yml). Got {yaml_config_file}")
    if not inputs_file.suffix == ".txt":
        parser.error(f"File with neural network inputs must be .txt file. Got {inputs_file}")
    if not outputs_file.suffix == ".txt":
        parser.error(f"File with neural network outputs must be .txt file. Got {outputs_file}")

    PROJECT_ROOT = Path(__file__).parent
    working_dir = os.path.join(PROJECT_ROOT, yaml_config_file.parent, "ray_results")

    nn_type = read_nn_type(yaml_config_file)
    experiment_name = f"tuning_{nn_type}"

    print(f"\nYAML config file:              {yaml_config_file}")
    print(f"Input list .txt file:          {inputs_file}")
    print(f"Output list .txt file:         {outputs_file}")
    print(f"Output variable index:         {var_index}")
    print(f"Tuning metric:                 {tuning_metric}")
    print(f"Experiment working directory:  {working_dir}")
    print(f"Restore directory:             {restore_directory}")

    print(f"\n\n{datetime.datetime.now()} --- Start tuning model {nn_type}.", flush=True)
    t_init = time.time()

    tune_model(yaml_config_file, inputs_file, outputs_file, var_index, tuning_search_alg, tuning_metric, search_space,
               random_seed, experiment_name=experiment_name, storage_path=working_dir, restore_dir=restore_directory)

    t_total = datetime.timedelta(seconds=time.time() - t_init)
    print(f"\n{datetime.datetime.now()} --- Finished. Elapsed time: {t_total}")
