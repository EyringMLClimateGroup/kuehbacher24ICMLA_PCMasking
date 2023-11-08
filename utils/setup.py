import getopt
import os
import sys
from pathlib import Path

import yaml
from scipy.stats import pearsonr
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.parcorr import ParCorr

from . import utils
from .constants import SIGNIFICANCE  # EXPERIMENT
from .constants import SPCAM_Vars, ANCIL_FILE  # DATA_FOLDER
from .variable import Variable_Lev_Metadata


class Setup:
    def __init__(self, argv):
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

        self.project_root = Path(__file__).parent.parent.resolve()

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
        if os.path.exists(path):
            return path
        elif Path(path).is_symlink():
            return os.path.realpath(path)
        else:
            return Path(self.project_root, path)


class SetupPCAnalysis(Setup):
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
        # Load specifications
        # self.analysis = yml_cfg["analysis"]
        # self.pc_alphas = yml_cfg["pc_alphas"]
        self.verbosity_pc = yml_cfg["verbosity"]
        # self.output_folder = yml_cfg["output_folder"]
        # self.output_file_pattern = yml_cfg["output_file_pattern"][self.analysis]
        # self.experiment = EXPERIMENT

        # region = yml_cfg["region"]
        # self.gridpoints = _calculate_gridpoints(region)

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

        # ## Level indexes (children & parents)
        # self.parents_idx_levs = [[lev, i] for i, lev in enumerate(self.levels)]  # All

        # lim_levels = yml_cfg["lim_levels"]
        # target_levels = yml_cfg["target_levels"]
        # target_levels = _calculate_target_levels(lim_levels, target_levels)
        # self.children_idx_levs = _calculate_children_level_indices(
        #     self.levels, target_levels, self.parents_idx_levs
        # )

        #  self.ind_test_name = yml_cfg["independence_test"]

        # Loaded here so errors are found during setup
        # Note the parenthesis, INDEPENDENCE_TESTS returns functions
        self.cond_ind_test = self.INDEPENDENCE_TESTS[self.ind_test_name]()

        self.overwrite_pc = yml_cfg.get("overwrite_pc", False)

        self.shifting = yml_cfg["shifting"]


class SetupPCMCIAggregation(Setup):
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
    def __init__(self, argv):
        super().__init__(argv)
        self._setup_neural_networks(self.yml_cfg)
        self._setup_neural_network_type(self.yml_cfg)
        self._setup_results_aggregation(self.yml_cfg)

    def _setup_neural_network_type(self, yml_cfg):
        self.nn_type = yml_cfg["nn_type"]

        # Set all possible types to False
        self.do_single_nn = False
        self.do_pca_nn = False
        self.do_causal_single_nn = False
        self.do_random_single_nn = False
        self.do_sklasso_nn = False

        # Set do_mirrored_strategy
        try:
            self.distribute_strategy = yml_cfg["distribute_strategy"]
        except KeyError:
            self.distribute_strategy = ""

        if self.nn_type == "SingleNN":
            self.do_single_nn = True

        elif self.nn_type == "pcaNN":
            self.do_pca_nn = True
            self.n_components = yml_cfg["n_components"]

        elif self.nn_type == "sklassoNN":
            self.do_sklasso_nn = True
            self.alpha_lasso = yml_cfg["alpha_lasso"]

        elif self.nn_type == "RandomSingleNN" or self.nn_type == "RandCorrSingleNN":
            self.do_random_single_nn = True

        elif self.nn_type == "CausalSingleNN" or self.nn_type == "CorrSingleNN":
            self.do_causal_single_nn = True

        elif self.nn_type == "CASTLEOriginal":
            self.beta = float(yml_cfg["beta"])
            self.lambda_weight = float(yml_cfg["lambda_weight"])
            self._set_common_castle_attributes(yml_cfg)

        elif self.nn_type == "CASTLEAdapted":
            self.lambda_prediction = float(yml_cfg["lambda_prediction"])
            self.lambda_sparsity = float(yml_cfg["lambda_sparsity"])
            self.lambda_acyclicity = float(yml_cfg["lambda_acyclicity"])
            self.lambda_reconstruction = float(yml_cfg["lambda_reconstruction"])
            self.acyclicity_constraint = yml_cfg["acyclicity_constraint"]
            self._set_common_castle_attributes(yml_cfg)

        elif self.nn_type == "castleNN":
            # Legacy version of CASTLE. Keep this for backwards compatibility
            self.rho = float(yml_cfg["rho"])
            self.alpha = float(yml_cfg["alpha"])
            self.beta = float(yml_cfg["beta"])
            self.lambda_weight = float(yml_cfg["lambda_weight"])
            self._set_additional_val_datasets(yml_cfg)

        elif self.nn_type == "all":
            self.do_single_nn = True
            self.do_causal_single_nn = True

        else:
            raise ValueError(f"Unknown Network type: {self.nn_type}")

    def _set_common_castle_attributes(self, yml_cfg):
        self.rho = float(yml_cfg["rho"])
        self.alpha = float(yml_cfg["alpha"])

        if yml_cfg["activation"].lower() == "leakyrelu":
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
        self.additional_val_datasets = yml_cfg.get("additional_val_datasets")
        for name_and_data in self.additional_val_datasets:
            data = self._evaluate_data_path(name_and_data['data'])
            if not os.path.exists(data):
                raise ValueError(f"Data path for additional dataset {name_and_data['data']} does not exist: "
                                 f"{name_and_data['name']}")
            name_and_data['data'] = data

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
        self.tensorboard_folder = yml_cfg["tensorboard_folder"]

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
                                "min_lr": yml_cfg["min_lr"]}  # 1e-8
        elif lr_schedule == "linear":
            self.lr_schedule = {"schedule": "linear",
                                "decay_steps": yml_cfg["decay_steps"],
                                "end_lr": yml_cfg["end_lr"]}
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
    def __init__(self, argv):
        super().__init__(argv)
        self._setup_diagnostics(self.yml_cfg)

    def _setup_diagnostics(self, yml_cfg):
        self.test_data_folder = self._evaluate_data_path(yml_cfg["test_data_folder"])
        self.test_data_fn = yml_cfg["test_data_fn"]
        self.diagnostics = yml_cfg["diagnostics"]
        self.diagnostics_time = yml_cfg["diagnostics_time"]


class SetupSherpa(SetupNeuralNetworks):
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
    # TODO? Move this out of here, to a utils module
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
