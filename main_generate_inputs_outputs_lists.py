import argparse
from pathlib import Path

from pcmasking.neural_networks.models_split_over_nodes import write_inputs_and_outputs_lists, write_outputs_mapping
from pcmasking.utils.setup import SetupNeuralNetworks


def create_inputs_outputs_files(config_file, inputs_file, outputs_file, mapping_file):

    argv = ["-c", config_file]
    setup = SetupNeuralNetworks(argv)

    write_inputs_and_outputs_lists(setup, inputs_file, outputs_file)
    write_outputs_mapping(setup, mapping_file)


if __name__ == "__main__":
    """
    Main function to generate text files for neural network input and output variables, as well as 
    output mappings from variable names to save strings, based on a given YAML configuration file.

    Command-line Arguments:
        -i, --out_file_inputs (str, optional): Path to the output .txt file for neural network inputs list. 
                                               Defaults to 'inputs_list.txt'.
        -o, --out_file_outputs (str, optional): Path to the output .txt file for neural network outputs list. 
                                                Defaults to 'outputs_list.txt'.
        -m, --out_file_mapping (str, optional): Path to the output .txt file for mapping neural network outputs 
                                                from variable names to save strings. Defaults to 'outputs_mapping.txt'.
        -c, --config_file (str, required): Path to the YAML configuration file for neural network creation.

    Variables:
        yaml_config_file (Path): Path object for the YAML configuration file.
        list_file_inputs (Path): Path object for the file where neural network input variables will be saved.
        list_file_outputs (Path): Path object for the file where neural network output variables will be saved.
        mapping_file_outputs (Path): Path object for the file where output variable mappings will be saved.

    Raises:
        ArgumentError: If the provided configuration file is not a YAML file or if the output files are not .txt files.

    Example:
        $ python create_inputs_outputs.py -c config.yml -i inputs.txt -o outputs.txt -m mapping.txt

    Workflow:
        1. Parse command-line arguments to get paths for the YAML configuration file and the output .txt files.
        2. Validate that the configuration file is in YAML format and the output files are in .txt format.
        3. Create directories for the output files if they do not exist.
        4. Call the `create_inputs_outputs_files` function to generate the .txt files for neural network input/output 
           variables and output mappings.
        5. The input/output variables and mappings are saved to the specified .txt files.
    """
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

    print(f"\nCreating .txt files for input and output variables for network training.", flush=True)
    print(f"Using config file: {yaml_config_file}\n", flush=True)

    create_inputs_outputs_files(yaml_config_file, list_file_inputs, list_file_outputs, mapping_file_outputs)
