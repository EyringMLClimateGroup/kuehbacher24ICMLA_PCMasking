import logging
import os
import unittest
from pathlib import Path

from neural_networks.load_models import load_models
from neural_networks.model_diagnostics import ModelDiagnostics
from notebooks_castle.test.testing_utils import set_memory_growth_gpu, train_model_if_not_exists
from utils.setup import SetupDiagnostics


# todo: functions to test
#  - get_shapley_values(self, itime, var, nTime=False, nSamples=False, metric=False)
#  - itime='range' for
#      plot_double_xy(self, itime, var, nTime=False, save=False, diff=False, stats=False, show_plot=True, **kwargs)
#  - plot_double_yz(self, var, varkeys, itime=1, nTime=False, ilon=1, save=False, diff=False, stats=False, **kwargs)
#  - get_profiles(self, var, varkeys, itime=1, nTime=False, lats=[-90, 90], lons=[0., 359.], stats=False, **kwargs)
#  - plot_double_profile(self, var, varkeys, itime=1, nTime=False, lats=[-90, 90], lons=[0., 359.], save=False, stats=False, **kwargs)
#  - compute_stats(self, itime, var, nTime=False)
#  - plot_profiles(vars_dict, varname='', title='', unit='', save=False, stats=False, **kwargs) not in model description

class TestCastleSetup(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)

        try:
            set_memory_growth_gpu()
        except RuntimeError:
            logging.warning("GPU growth could not be enabled. "
                            "When running multiple tests, this may be because the physical drivers are already "
                            "initialized, in which case memory growth may already be enabled. "
                            "If memory growth is not enabled, the tests may fail with CUDA error.")

        # Just two 2d variables, which results in 2 single output networks
        argv = ["-c", "cfg_castle_NN_Creation_test_2.yml"]

        self.castle_setup = SetupDiagnostics(argv)
        train_model_if_not_exists(self.castle_setup)

        self.castle_models = load_models(self.castle_setup)

        self.castle_setup.model_type = self.castle_setup.nn_type

        self.plots_save_dir = os.path.join(Path(__file__).parent.resolve(), "output", "plots")

    def test_creating_model_diagnostic(self):
        logging.info("Testing creating ModelDiagnostics instance for loaded CASTLE models. ")
        castle_md = ModelDiagnostics(setup=self.castle_setup,
                                     models=self.castle_models[self.castle_setup.nn_type])
        print(castle_md)

    def test_plot_plot_double_xy_itime_int(self):
        castle_md = ModelDiagnostics(setup=self.castle_setup,
                                     models=self.castle_models[self.castle_setup.nn_type])
        itime = 1
        self._plot_diff_true_false(castle_md, itime)

    def test_plot_plot_double_xy_itime_mean(self):
        castle_md = ModelDiagnostics(setup=self.castle_setup,
                                     models=self.castle_models[self.castle_setup.nn_type])
        itime = 'mean'
        self._plot_diff_true_false(castle_md, itime)

    # todo: plot_double_xy with range is not working
    # def test_plot_plot_double_xy_itime_range(self):
    #     castle_md = ModelDiagnostics(setup=self.castle_setup,
    #                                  models=self.castle_models[self.castle_setup.nn_type])
    #     itime = 'range'
    #     self._plot_diff_true_false(castle_md, itime)

    # todo: adjust shapes from train_gen
    # def test_get_shapley_values(self):
    #     castle_md = ModelDiagnostics(setup=self.castle_setup,
    #                                  models=self.castle_models[self.castle_setup.nn_type])
    #     itime = 'range'
    #     for var in self.castle_models[self.castle_setup.nn_type].keys():
    #         print(var)
    #         castle_md.get_shapley_values(itime, var, nTime=False, nSamples=False, metric=False)
    #
    # def test_compute_stats(self):
    #     castle_md = ModelDiagnostics(setup=self.castle_setup,
    #                                  models=self.castle_models[self.castle_setup.nn_type])
    #
    #     itime = 'range'
    #     for var in self.castle_models[self.castle_setup.nn_type].keys():
    #         # Todo: do I have to call compute before mean?
    #         castle_md.compute_stats(itime, var)
    #         stats = castle_md.mean_stats()
    #         print(f"Stats for variable {var}: \n{stats}\n")

    def _plot_diff_true_false(self, model_diagnostic, itime):
        for var in self.castle_models[self.castle_setup.nn_type].keys():
            print(var)
            model_diagnostic.plot_double_xy(itime, var, diff=False, nTime=False, cmap='RdBu_r',
                                            save=self.plots_save_dir)

        for var in self.castle_models[self.castle_setup.nn_type].keys():
            print(var)
            model_diagnostic.plot_double_xy(itime, var, diff=True, nTime=False, cmap='RdBu_r', save=self.plots_save_dir)
