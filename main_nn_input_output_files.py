import argparse
from pathlib import Path

from neural_networks.models_split_over_nodes import generate_inputs_and_outputs
from utils.setup import SetupNeuralNetworks


def create_inputs_outputs_files(config_file, inputs_file, outputs_file):
    argv = ["-c", config_file]
    setup = SetupNeuralNetworks(argv)

    generate_inputs_and_outputs(setup, inputs_file, outputs_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates .txt files for neural network input and output "
                                                 "variables for specific setup configuration.")
    parser.add_argument("-c", "--config_file", help="YAML configuration file for neural network creation.")
    parser.add_argument("-i", "--out_file_inputs", default="inputs_list.txt",
                        help=".txt output file for NN inputs list.")
    parser.add_argument("-o", "--out_file_outputs", default="outputs_list.txt",
                        help=".txt output file for NN outputs list.")

    args = parser.parse_args()

    yaml_config_file = Path(args.config_file)
    out_file_inputs = Path(args.out_file_inputs)
    out_file_outputs = Path(args.out_file_outputs)

    if not yaml_config_file.suffix == ".yml":
        parser.error(f"Configuration file must be YAML file (.yml). Got {yaml_config_file}")
    if not out_file_inputs.suffix == ".txt":
        parser.error(f"Output file for inputs must be .txt file. Got {out_file_inputs}")
    if not out_file_outputs.suffix == ".txt":
        parser.error(f"Output file for outputs must be .txt file. Got {out_file_outputs}")

    Path(out_file_inputs.parent).mkdir(parents=True, exist_ok=True)
    Path(out_file_outputs.parent).mkdir(parents=True, exist_ok=True)

    print(f"Creating .txt files for input and output variables for CASTLE training.", flush=True)

    create_inputs_outputs_files(yaml_config_file, out_file_inputs, out_file_outputs)
