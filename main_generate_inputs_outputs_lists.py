import argparse
from pathlib import Path

from neural_networks.models_split_over_nodes import write_inputs_and_outputs_lists, write_outputs_mapping
from utils.setup import SetupNeuralNetworks


def create_inputs_outputs_files(config_file, inputs_file, outputs_file, mapping_file):
    argv = ["-c", config_file]
    setup = SetupNeuralNetworks(argv)

    write_inputs_and_outputs_lists(setup, inputs_file, outputs_file)
    write_outputs_mapping(setup, mapping_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates .txt files for neural network input and output "
                                                 "variables for specific setup configuration.")

    parser.add_argument("-i", "--out_file_inputs", nargs="?", default="inputs_list.txt",
                        help=".txt output file for NN inputs list.")
    parser.add_argument("-o", "--out_file_outputs", nargs="?", default="outputs_list.txt",
                        help=".txt output file for NN outputs list.")
    parser.add_argument("-m", "--out_file_mapping", nargs="?", default="outputs_mapping.txt",
                        help=".txt output file for NN outputs mapping from variable name to save string.")

    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("-c", "--config_file", help="YAML configuration file for neural network creation.",
                               required=True)

    args = parser.parse_args()

    yaml_config_file = Path(args.config_file)
    list_file_inputs = Path(args.out_file_inputs)
    list_file_outputs = Path(args.out_file_outputs)
    mapping_file_outputs = Path(args.out_file_mapping)

    if not yaml_config_file.suffix == ".yml":
        parser.error(f"Configuration file must be YAML file (.yml). Got {yaml_config_file}")
    if not list_file_inputs.suffix == ".txt":
        parser.error(f"Output file for inputs list must be .txt file. Got {list_file_inputs}")
    if not list_file_outputs.suffix == ".txt":
        parser.error(f"Output file for outputs list must be .txt file. Got {list_file_outputs}")
    if not mapping_file_outputs.suffix == ".txt":
        parser.error(f"Output file for outputs mapping must be .txt file. Got {mapping_file_outputs}")

    Path(list_file_inputs.parent).mkdir(parents=True, exist_ok=True)
    Path(list_file_outputs.parent).mkdir(parents=True, exist_ok=True)
    Path(mapping_file_outputs.parent).mkdir(parents=True, exist_ok=True)

    print(f"\nCreating .txt files for input and output variables for CASTLE training.", flush=True)
    print(f"Using config file: {yaml_config_file}\n", flush=True)

    create_inputs_outputs_files(yaml_config_file, list_file_inputs, list_file_outputs, mapping_file_outputs)
