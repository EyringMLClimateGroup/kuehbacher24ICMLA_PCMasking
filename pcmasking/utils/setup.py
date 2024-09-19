import getopt
import os
import sys
from pathlib import Path

import yaml
from scipy.stats import pearsonr
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.parcorr import ParCorr

from pcmasking.utils import utils
from pcmasking.utils.constants import SIGNIFICANCE  # EXPERIMENT
from pcmasking.utils.constants import SPCAM_Vars, ANCIL_FILE  # DATA_FOLDER
from pcmasking.utils.variable import Variable_Lev_Metadata


class Setup:
    """Base setup class for configuration and environment initialization.

    Attributes:
        project_root (Path): Root directory of the project.
        yml_filename (str): Path to the YAML configuration file.
        yml_cfg (dict): Parsed YAML configuration data.
        analysis (str): Type of analysis specified in the YAML file.
        pc_alphas (list): List of PC1 alphas.
        output_folder (str): Path to the output folder.
        output_file_pattern (str): Pattern for the output file names.
        experiment (str): Experiment details from the YAML config.
        data_folder (Path): Path to the data folder.
        region (str): Geographic region of interest.
        gridpoints (list): List of grid points for analysis.
        levels (list): Atmospheric levels.
        parents_idx_levs (list): Parent variable levels.
        target_levels (list): Target atmospheric levels.
        children_idx_levs (list): Indices of child variable levels.
        spcam_inputs (list): List of SPCAM input variables.
        spcam_outputs (list): List of SPCAM output variables.
        ind_test_name (str): Name of the independence test used in the analysis.
    """

    def __init__(self, argv):
        """
        Initializes the Setup object by parsing command-line arguments and loading the configuration file.

        Args:
            argv (list): List of command-line arguments with the configuration file.
        """
        try:
            opts, args = getopt.getopt(argv, "hc:a", ["cfg_file=", "add="])
        except getopt.GetoptError:
            print("pipeline.py -c [cfg_file] -a [add]")
            sys.exit(2)
        for opt, arg in opts:
            if opt == "-h":
                print("pipeline.py -c [cfg_file]")
                sys.exit()
            elif opt in ("-c", "--cfg_file"):
                yml_cfgFilenm = arg
            elif opt in ("-a", "--add"):
                pass

        self.project_root = Path(__file__).parent.parent.parent.resolve()

        # YAML config file
        self.yml_filename = yml_cfgFilenm
        with open(self.yml_filename, "r") as yml_cfgFile:
            self.yml_cfg = yaml.load(yml_cfgFile, Loader=yaml.FullLoader)

        self._setup_common(self.yml_cfg)

    def _setup_common(self, yml_cfg):
        # Load specifications
        self.analysis = yml_cfg["analysis"]
        self.pc_alphas = yml_cfg["pc_alphas"]
        #         self.verbosity = yml_cfg["verbosity"]
        self.output_folder = yml_cfg["output_folder"]
        self.output_file_pattern = yml_cfg["output_file_pattern"][self.analysis]
        self.experiment = yml_cfg["experiment"]
        self.data_folder = self._evaluate_data_path(yml_cfg["data_folder"])

        self.region = yml_cfg["region"]
        self.gridpoints = _calculate_gridpoints(self.region)

        ## Model's grid
        self.levels, latitudes, longitudes = utils.read_ancilaries(
            Path(ANCIL_FILE)
        )

        ## Level indexes (children & parents)
        self.parents_idx_levs = [[lev, i] for i, lev in enumerate(self.levels)]  # All

        lim_levels = yml_cfg["lim_levels"]
        self.target_levels = yml_cfg["target_levels"]
        target_levels = _calculate_target_levels(lim_levels, self.target_levels)
        self.children_idx_levs = _calculate_children_level_indices(
            self.levels, target_levels, self.parents_idx_levs
        )

        ## Variables
        spcam_parents = yml_cfg["spcam_parents"]
        spcam_children = yml_cfg["spcam_children"]
        self.list_spcam = [
            var for var in SPCAM_Vars if var.name in spcam_parents + spcam_children
        ]
        self.spcam_inputs = [var for var in self.list_spcam if var.type == "in"]
        self.spcam_outputs = [var for var in self.list_spcam if var.type == "out"]

        self.ind_test_name = yml_cfg["independence_test"]

        # # Loaded here so errors are found during setup
        # # Note the parenthesis, INDEPENDENCE_TESTS returns functions
        # self.cond_ind_test = INDEPENDENCE_TESTS[self.ind_test_name]()

    def _evaluate_data_path(self, path):
        if os.path.isabs(path):
            return Path(path)
        elif Path(path).is_symlink():
            return Path(os.path.realpath(path))
        else:
            return Path(self.project_root, path)


class SetupPCAnalysis(Setup):
    """
    Setup class for configuring PC1 analysis.

    Inherits from Setup.

    Attributes:
        verbosity_pc (int): Verbosity level for P1C analysis.
        idx_lats (list): Latitude indices for grid points.
        idx_lons (list): Longitude indices for grid points.
        cond_ind_test (function): Conditional independence test function.
        overwrite_pc (bool): Whether to overwrite existing PC1 analysis results.
        shifting (bool): Whether shifting is enabled in the analysis.
    """
    INDEPENDENCE_TESTS = {
        "parcorr": lambda: ParCorr(significance=SIGNIFICANCE),
        "gpdc": lambda: GPDC(recycle_residuals=True),
        "gpdc_torch": lambda: _build_GPDCtorch(recycle_residuals=True),
        # "gpdc_torch" : lambda: _build_GPDCtorch(recycle_residuals=False),
        "pearsonr": lambda: pearsonr,
    }

    def __init__(self, argv):
        super().__init__(argv)
        self._setup_pc_analysis(self.yml_cfg)

    def _setup_pc_analysis(self, yml_cfg):
        self.verbosity_pc = yml_cfg["verbosity"]

        ## Model's grid
        self.levels, latitudes, longitudes = utils.read_ancilaries(
            Path(ANCIL_FILE)
        )

        ## Latitude / Longitude indexes
        self.idx_lats = [
            utils.find_closest_value(latitudes, gridpoint[0])
            for gridpoint in self.gridpoints
        ]
        self.idx_lons = [
            utils.find_closest_longitude(longitudes, gridpoint[1])
            for gridpoint in self.gridpoints
        ]

        # Loaded here so errors are found during setup
        # Note the parenthesis, INDEPENDENCE_TESTS returns functions
        self.cond_ind_test = self.INDEPENDENCE_TESTS[self.ind_test_name]()

        self.overwrite_pc = yml_cfg.get("overwrite_pc", False)

        self.shifting = yml_cfg["shifting"]


class SetupPCMCIAggregation(Setup):
    """Setup class for configuring results aggregation and plotting for PCMCI analysis.

    Inherits from Setup.

    Attributes:
        thresholds (list): Threshold values used for aggregation.
        area_weighted (bool): Whether area weighting is applied.
        pdf (bool): Whether to generate PDFs.
        aggregate_folder (Path): Path to the folder for storing aggregation results.
        plots_folder (str): Path to the folder for storing plots.
        plot_file_pattern (str): File pattern for saving plot images.
        overwrite_plots (bool): Whether to overwrite existing plot files.
    """

    def __init__(self, argv):
        super().__init__(argv)
        self._setup_results_aggregation(self.yml_cfg)
        self._setup_plots(self.yml_cfg)

    def _setup_results_aggregation(self, yml_cfg):
        self.thresholds = yml_cfg["thresholds"]
        self.area_weighted = yml_cfg["area_weighted"]
        self.pdf = yml_cfg["pdf"]
        self.aggregate_folder = self._evaluate_data_path(yml_cfg["aggregate_folder"])

    def _setup_plots(self, yml_cfg):
        self.plots_folder = yml_cfg["plots_folder"]
        self.plot_file_pattern = yml_cfg["plot_file_pattern"][self.analysis]
        self.overwrite_plots = yml_cfg.get("overwrite_plot", False)


class SetupNeuralNetworks(Setup):
    """Setup class for configuring neural network training and evaluation.

    Inherits from Setup.

    Attributes:
        nn_type (str): Type of neural network being trained.
        do_single_nn (bool): Whether to train a single neural network.
        do_causal_single_nn (bool): Whether to train a causal single neural network.
        do_random_single_nn (bool): Whether to train a random single neural network.
        distribute_strategy (str): Distribution strategy for parallel training.
        hidden_layers (list): List of hidden layers units.
        activation (str): Activation function used in the network.
        epochs (int): Number of epochs for training.
        train_verbose (int): Verbosity level during training.
        tensorboard_folder (Path): Path for saving TensorBoard logs.
        train_data_folder (Path): Path to the training data folder.
        normalization_folder (Path): Path to the normalization folder.
        init_lr (float): Initial learning rate for training.
        lr_schedule (dict): Learning rate schedule configuration.
        train_patience (int): Patience for early stopping during training.
    """

    def __init__(self, argv):
        super().__init__(argv)
        self._setup_neural_networks(self.yml_cfg)
        self._setup_neural_network_type(self.yml_cfg)
        self._setup_results_aggregation(self.yml_cfg)

    def _setup_neural_network_type(self, yml_cfg):
        self.nn_type = yml_cfg["nn_type"]

        # Set all possible types to False
        self.do_single_nn = False
        self.do_causal_single_nn = False
        self.do_random_single_nn = False

        # Set do_mirrored_strategy
        try:
            self.distribute_strategy = yml_cfg["distribute_strategy"]
        except KeyError:
            self.distribute_strategy = ""

        if self.nn_type == "SingleNN":
            self.do_single_nn = True

        elif self.nn_type == "RandomSingleNN" or self.nn_type == "RandCorrSingleNN":
            self.do_random_single_nn = True

        elif self.nn_type == "CausalSingleNN" or self.nn_type == "CorrSingleNN":
            self.do_causal_single_nn = True


        elif self.nn_type == "PreMaskNet":
            self.lambda_sparsity = float(yml_cfg["lambda_sparsity"])
            self._set_common_pcmasking_attributes(yml_cfg)

        elif self.nn_type == "MaskNet":
            try:
                self.mask_threshold = float(yml_cfg["mask_threshold"])
                self.mask_threshold_file = None
            except KeyError:
                self.mask_threshold_file = self._evaluate_data_path(yml_cfg["mask_threshold_file"])
                self.mask_threshold = None

            self.masking_vector_file = self._evaluate_data_path(yml_cfg["masking_vector_file"])
            if not self.masking_vector_file.name.endswith(".npy"):
                raise ValueError(f"Expected masking vector to be saved in numpy format .npy. "
                                 f"Got file {self.masking_vector_file}")

            self._set_common_pcmasking_attributes(yml_cfg)


        elif self.nn_type == "all":
            self.do_single_nn = True
            self.do_causal_single_nn = True

        else:
            raise ValueError(f"Unknown Network type: {self.nn_type}")

    def _set_common_pcmasking_attributes(self, yml_cfg):
        try:
            self.relu_alpha = float(yml_cfg["relu_alpha"])
        except KeyError:
            self.relu_alpha = 0.3

        self._set_additional_val_datasets(yml_cfg)

        kernel_initializer_input_layers = yml_cfg.get("kernel_initializer_input_layers")
        self.kernel_initializer_input_layers = _set_initializer_params(kernel_initializer_input_layers,
                                                                       "input_", yml_cfg)

        kernel_initializer_hidden_layers = yml_cfg.get("kernel_initializer_hidden_layers")
        self.kernel_initializer_hidden_layers = _set_initializer_params(
            kernel_initializer_hidden_layers, "hidden_", yml_cfg)

        kernel_initializer_output_layers = yml_cfg.get("kernel_initializer_output_layers")
        self.kernel_initializer_output_layers = _set_initializer_params(
            kernel_initializer_output_layers, "output_", yml_cfg)

    def _set_additional_val_datasets(self, yml_cfg):
        try:
            self.additional_val_datasets = yml_cfg["additional_val_datasets"]

            for name_and_data in self.additional_val_datasets:
                data = self._evaluate_data_path(name_and_data['data'])
                if not os.path.exists(data):
                    raise ValueError(f"Data path for additional dataset {name_and_data['data']} does not exist: "
                                     f"{name_and_data['name']}")
                name_and_data['data'] = data
        except KeyError:
            # No additional validation datasets were given
            pass

    def _setup_results_aggregation(self, yml_cfg):
        self.thresholds = yml_cfg["thresholds"]
        self.area_weighted = yml_cfg["area_weighted"]
        self.pdf = yml_cfg["pdf"]
        self.aggregate_folder = self._evaluate_data_path(yml_cfg["aggregate_folder"])

    def _setup_neural_networks(self, yml_cfg):
        self.nn_output_path = self._evaluate_data_path(yml_cfg["nn_output_path"])

        input_order = yml_cfg["input_order"]
        self.input_order = [
            SPCAM_Vars[x] for x in input_order if SPCAM_Vars[x].type == "in"
        ]
        self.input_order_list = _make_order_list(self.input_order, self.levels)
        output_order = yml_cfg["output_order"]
        self.output_order = [
            SPCAM_Vars[x] for x in output_order if SPCAM_Vars[x].type == "out"
        ]
        self.hidden_layers = yml_cfg["hidden_layers"]
        self.activation = yml_cfg["activation"]
        self.epochs = yml_cfg["epochs"]

        # Training configuration
        self.train_verbose = yml_cfg["train_verbose"]
        self.tensorboard_folder = self._evaluate_data_path(yml_cfg["tensorboard_folder"])

        self.train_data_folder = self._evaluate_data_path(yml_cfg["train_data_folder"])
        self.train_data_fn = yml_cfg["train_data_fn"]
        self.valid_data_fn = yml_cfg["valid_data_fn"]

        self.normalization_folder = self._evaluate_data_path(yml_cfg["normalization_folder"])
        self.normalization_fn = yml_cfg["normalization_fn"]

        self.input_sub = yml_cfg["input_sub"]
        self.input_div = yml_cfg["input_div"]
        self.out_scale_dict_folder = self._evaluate_data_path(yml_cfg["out_scale_dict_folder"])
        self.out_scale_dict_fn = yml_cfg["out_scale_dict_fn"]
        self.batch_size = yml_cfg["batch_size"]

        # Add an attributed for validation batch size
        # Even though it's not used in the normal training, it's good to be able to control
        #  validation batch size in testing
        # Using get here, so that it doesn't throw a key not found error if validation batch size was
        #  not specified in the config file
        self.val_batch_size = yml_cfg.get("val_batch_size")
        self.use_val_batch_size = yml_cfg.get("val_batch_size")

        # Learning rate
        self.init_lr = yml_cfg["init_lr"]
        # Learning rate schedule
        lr_schedule = yml_cfg.get("lr_schedule")

        self._set_learning_rate_schedule(lr_schedule, yml_cfg)

        self.train_patience = yml_cfg["train_patience"]

    def _set_learning_rate_schedule(self, lr_schedule, yml_cfg):
        # Backwards compatibility for config files that don't explicitly specify a schedule
        # In this case, exponential schedule is assumed
        if lr_schedule is None or lr_schedule == "exponential":
            self.lr_schedule = {"schedule": "exponential",
                                "step": yml_cfg["step_lr"],
                                "divide": yml_cfg["divide_lr"]}

            # Also keep the old attributes to ensure compatibility with old code parts
            self.step_lr = yml_cfg["step_lr"]
            self.divide_lr = yml_cfg["divide_lr"]
        elif lr_schedule == "plateau":
            self.lr_schedule = {"schedule": "plateau",
                                "monitor": yml_cfg["monitor"],  # val_loss
                                "factor": float(yml_cfg["factor"]),
                                "patience": yml_cfg["patience"],
                                "min_lr": float(yml_cfg["min_lr"])}  # 1e-8
        elif lr_schedule == "linear":
            self.lr_schedule = {"schedule": "linear",
                                "decay_steps": yml_cfg["decay_steps"],
                                "end_lr": float(yml_cfg["end_lr"])}
        elif lr_schedule == "cosine":
            self.lr_schedule = {"schedule": "cosine",
                                "decay_steps": yml_cfg["decay_steps"],
                                "alpha": yml_cfg["cosine_alpha"],
                                "warmup_steps": yml_cfg["warmup_steps"]}


def _set_initializer_params(initializer, prefix, yml_cfg):
    if initializer is None:
        return None
    elif initializer == "Constant":
        try:
            kernel_initializer = {"initializer": initializer,
                                  "value": float(yml_cfg[prefix + "init_constant_value"])}
        except KeyError:
            raise ValueError(
                "Missing value for 'value' parameter for tf.keras.initializers.Constant initializer.")
    elif initializer == "Identity":
        try:
            kernel_initializer = {"initializer": initializer,
                                  "gain": float(yml_cfg[prefix + "init_identity_gain"])}
        except KeyError:
            raise ValueError(
                "Missing value for 'gain' parameter for tf.keras.initializers.Identity initializer.")
    elif initializer == "Orthogonal":
        try:
            kernel_initializer = {"initializer": initializer,
                                  "gain": float(yml_cfg[prefix + "init_orthogonal_gain"])}
        except KeyError:
            raise ValueError(
                "Missing value for 'gain' parameter for tf.keras.initializers.Orthogonal initializer.")
    elif initializer == "RandomNormal":
        try:
            kernel_initializer = {"initializer": initializer,
                                  "mean": float(yml_cfg[prefix + "init_random_normal_mean"]),
                                  "std": float(yml_cfg[prefix + "init_random_normal_std"])}
        except KeyError:
            raise ValueError(
                "Missing value for 'mean' or 'std' parameter for tf.keras.initializers.RandomNormal initializer.")
    elif initializer == "RandomUniform":
        try:
            kernel_initializer = {"initializer": initializer,
                                  "min_val": float(yml_cfg[prefix + "init_random_uniform_min_val"]),
                                  "max_val": float(yml_cfg[prefix + "init_random_uniform_max_val"])}
        except KeyError:
            raise ValueError(
                "Missing value for 'min_val' or 'max_val' parameter for tf.keras.initializers.RandomUniform initializer.")
    elif initializer == "TruncatedNormal":
        try:
            kernel_initializer = {"initializer": initializer,
                                  "mean": float(yml_cfg[prefix + "init_truncated_normal_mean"]),
                                  "std": float(yml_cfg[prefix + "init_truncated_normal_std"])}
        except KeyError:
            raise ValueError(
                "Missing value for 'mean' or 'std' parameter for tf.keras.initializers.TruncatedNormal initializer.")
    elif initializer == "VarianceScaling":
        try:
            kernel_initializer = {"initializer": initializer,
                                  "scale": float(yml_cfg[prefix + "init_variance_scaling_scale"]),
                                  "mode": yml_cfg[prefix + "init_variance_scaling_mode"],
                                  "distribution": yml_cfg[prefix + "init_variance_scaling_distribution"]}

        except KeyError:
            raise ValueError(
                "Missing value for 'mean' or 'std' parameter for tf.keras.initializers.VarianceScaling initializer.")
    elif initializer in ["GlorotNormal", "GlorotUniform", "HeNormal",
                         "HeUniform", "LecunNormal", "LecunUniform",
                         "Ones", "Zeros"]:
        kernel_initializer = {"initializer": initializer}
    else:
        raise ValueError(f"Unknown {prefix} kernel initializer value: {initializer}. Possible values are "
                         f"['Constant', 'GlorotNormal', 'GlorotUniform', 'HeNormal', 'HeUniform', 'Identity', "
                         f"'LecunNormal', 'LecunUniform', 'Ones', 'Orthogonal', 'RandomNormal', 'RandomUniform', "
                         f"'TruncatedNormal', 'VarianceScaling', 'Zeros'].")
    return kernel_initializer


class SetupDiagnostics(SetupNeuralNetworks):
    """Setup class for configuring evaluation for neural networks.

    Inherits from SetupNeuralNetworks.

    Attributes:
        test_data_folder (Path): Path to the test data folder.
        test_data_fn (str): Test data filename.
        diagnostics (dict): Dictionary containing diagnostic configurations.
        diagnostics_time (str): Time configuration for diagnostics.
    """

    def __init__(self, argv):
        super().__init__(argv)
        self._setup_diagnostics(self.yml_cfg)

    def _setup_diagnostics(self, yml_cfg):
        self.test_data_folder = self._evaluate_data_path(yml_cfg["test_data_folder"])
        self.test_data_fn = yml_cfg["test_data_fn"]
        self.diagnostics = yml_cfg["diagnostics"]
        self.diagnostics_time = yml_cfg["diagnostics_time"]


class SetupSherpa(SetupNeuralNetworks):
    """Setup class for configuring Sherpa hyperparameter optimization for neural networks.

    Inherits from SetupNeuralNetworks.

    Attributes:
        sherpa_hyper (dict): Hyperparameter tuning settings for Sherpa.
        nn_sherpa_path (str): Path to the neural network Sherpa experiment.
        sherpa_pc_alphas (list): List of PC alphas for Sherpa experiments.
        sherpa_thresholds (list): Thresholds for Sherpa experiments.
        sherpa_num_layers (int): Number of layers in the Sherpa-tuned neural network.
        sherpa_num_nodes (int): Number of nodes per layer in the Sherpa-tuned neural network.
        sherpa_num_trials (int): Number of Sherpa trials for tuning.
    """

    def __init__(self, argv):
        super().__init__(argv)
        self._setup_sherpa(self.yml_cfg)

    def _setup_sherpa(self, yml_cfg):
        self.sherpa_hyper = yml_cfg["sherpa_hyper"]
        self.nn_type = yml_cfg["sherpa_nn_type"]
        self.nn_sherpa_path = yml_cfg["nn_sherpa_path"]
        self.sherpa_pc_alphas = yml_cfg["sherpa_pc_alphas"]
        self.sherpa_thresholds = yml_cfg["sherpa_thresholds"]
        self.sherpa_num_layers = yml_cfg["sherpa_num_layers"]
        self.sherpa_num_nodes = yml_cfg["sherpa_num_nodes"]
        self.sherpa_num_trials = yml_cfg["sherpa_num_trials"]


def _calculate_gridpoints(region):
    ## Region / Gridpoints
    if region is False:
        region = [[-90, 90], [0, -0.5]]  # All
    return utils.get_gridpoints(region)


def _calculate_target_levels(lim_levels, target_levels):
    ## Children levels (parents includes all)
    if lim_levels is not False and target_levels is False:
        target_levels = utils.get_levels(lim_levels)
    return target_levels


def _calculate_children_level_indices(levels, target_levels, parents_idx_levs):
    if target_levels is not False:
        children_idx_levs = [
            [lev, utils.find_closest_value(levels, lev)] for lev in target_levels
        ]
    else:
        children_idx_levs = parents_idx_levs
    return children_idx_levs


def _build_GPDCtorch(**kwargs):
    """
    Helper function to isolate the GPDCtorch import, as it's not
    present in the master version of Tigramite
    """
    from tigramite.independence_tests import GPDCtorch

    return GPDCtorch(**kwargs)


def _make_order_list(order, levels):
    order_list = list()
    for i_var, spcam_var in enumerate(order):
        if spcam_var.dimensions == 3:
            n_levels = len(levels)
        elif spcam_var.dimensions == 2:
            n_levels = 1
        for i_lvl in range(n_levels):
            if spcam_var.dimensions == 3:
                level = levels[i_lvl]
                var = Variable_Lev_Metadata(spcam_var, level, i_lvl)
            elif spcam_var.dimensions == 2:
                var = Variable_Lev_Metadata(spcam_var, None, None)
            order_list.append(var)
    return order_list
