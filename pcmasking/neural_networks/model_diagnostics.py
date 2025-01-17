import gc
import os
import pickle
from copy import deepcopy
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
from ipykernel.kernelapp import IPKernelApp
from math import pi

from pcmasking.neural_networks.cbrain.cam_constants import P
from pcmasking.neural_networks.data_generator import build_train_generator, build_valid_generator
from pcmasking.neural_networks.load_models import get_var_list
from pcmasking.utils.constants import ANCIL_FILE
from pcmasking.utils.utils import read_ancilaries, find_closest_value, find_closest_longitude
from pcmasking.utils.variable import Variable_Lev_Metadata


def in_notebook():
    return IPKernelApp.initialized()


if in_notebook():
    pass
else:
    pass
import shap

cThemes = {'tphystnd': 'coolwarm',
           'phq': 'RdBu',
           'fsns': 'Reds',
           'flns': 'Reds',
           'fsnt': 'Reds',
           'flnt': 'Reds',
           'prect': 'PuBu'}


class ModelDiagnostics():
    """
    Class for evaluating of SPCAM parameterization models and generating various plots and statistics
    related to model predictions and SPCAM truth.

    Attributes:
        setup (pcmasking.utils.setup.Setup): Setup configuration object with model parameters.
        models (dict): Dictionary containing trained models for different variables.
        nlat (int): Number of latitude points.
        nlon (int): Number of longitude points.
        nlev (int): Number of vertical levels.
        ntime (int): Number of time steps.
        ngeo (int): Total number of grid points (nlat * nlon).
        levels (list): List of vertical levels.
        latitudes (list): List of latitude points.
        longitudes (list): List of longitude points.
        lat_weights (ndarray): Latitude weights for averaging based on cosine of latitude.
    """

    def __init__(self, setup, models, nlat=64, nlon=128, nlev=30, ntime=48):
        """
        Initializes the ModelDiagnostics class.

        Args:
            setup (pcmasking.utils.setup.Setup): Setup configuration object with model parameters.
            models (dict): Dictionary containing trained models for different variables.
            nlat (int, optional): Number of latitude points. Defaults to 64.
            nlon (int, optional): Number of longitude points. Defaults to 128.
            nlev (int, optional): Number of vertical levels. Defaults to 30.
            ntime (int, optional): Number of time steps. Defaults to 48.
        """
        self.nlat, self.nlon, self.nlev = nlat, nlon, nlev
        self.ngeo = nlat * nlon
        self.setup = setup
        self.models = models

        self.levels, self.latitudes, self.longitudes = read_ancilaries(
            Path(ANCIL_FILE)
        )
        self.lat_weights = np.cos(self.latitudes * pi / 180.)

    def reshape_ngeo(self, x, nTime=False):
        """Reshapes the input data to the correct geographic dimensions"""
        if nTime:
            return x.reshape(nTime, self.nlat, self.nlon)
        else:
            return x.reshape(self.nlat, self.nlon)

    def get_output_var_idx(self, var):
        """Gets the index of a specific output variable"""
        var_idxs = self.valid_gen.norm_ds.var_names[self.valid_gen.output_idxs]
        var_idxs = np.where(var_idxs == var)[0]
        return var_idxs

    def get_truth_pred(self, itime, var, nTime=False):
        """
        Retrieves the truth and prediction values for a specified variable at a given time step.

        Args:
            itime (int or str): The time step (int) or 'mean'/'range' (str) for averaged data.
            var (str): The name of the output variable.
            nTime (bool, optional): Whether the data includes a time dimension. Defaults to False.

        Returns:
            tuple: Tuple containing truth and prediction arrays.
        """
        input_list = get_var_list(self.setup, self.setup.spcam_inputs)
        self.inputs = sorted(
            [Variable_Lev_Metadata.parse_var_name(p) for p in input_list],
            key=lambda x: self.setup.input_order_list.index(x),
        )
        self.input_vars_dict = ModelDiagnostics._build_vars_dict(self.inputs)

        self.output = Variable_Lev_Metadata.parse_var_name(var)
        self.output_vars_dict = ModelDiagnostics._build_vars_dict([self.output])

        self.valid_gen = build_valid_generator(
            self.input_vars_dict,
            self.output_vars_dict,
            self.setup,
            test=True,
            diagnostic_mode=True
        )
        with self.valid_gen as valid_gen:
            model, inputs = self.models[var]

            if isinstance(itime, int):
                X, truth = valid_gen[itime]
                pred = model.predict_on_batch(X[:, inputs])

                # Inverse transform
                truth = valid_gen.output_transform.inverse_transform(truth)
                pred = valid_gen.output_transform.inverse_transform(pred)

            elif itime == 'mean' or itime == 'range':
                if not nTime:
                    nTime = len(self.valid_gen)
                print(f"Time samples: {nTime}")

                truth = np.zeros([nTime, self.ngeo, 1])
                pred = np.zeros([nTime, self.ngeo, 1])

                for iTime in range(nTime):
                    X_tmp, t_tmp = valid_gen[iTime]
                    p_tmp = model.predict_on_batch(X_tmp[:, inputs])

                    # Inverse transform
                    truth[iTime, :] = valid_gen.output_transform.inverse_transform(t_tmp)
                    pred[iTime, :] = valid_gen.output_transform.inverse_transform(p_tmp)
                if itime == 'mean':
                    truth = np.mean(truth, axis=0)
                    pred = np.mean(pred, axis=0)

        if itime == 'range':
            truth = self.reshape_ngeo(truth[:, :, 0], nTime=nTime)
            pred = self.reshape_ngeo(pred[:, :, 0], nTime=nTime)
        else:
            truth = self.reshape_ngeo(truth[:, 0])
            pred = self.reshape_ngeo(pred[:, 0])

        return truth, pred

    def get_shapley_values(
            self,
            itime,
            var,
            nTime=False,
            nSamples=False,
            metric=False
    ):
        """
        Computes SHAP values for a specific variable using shap.DeepExplainer.

        Args:
            itime (int or str): The time step (int) or 'range' (str) for averaged data.
            var (str): The name of the output variable.
            nTime (bool, optional): Number of time steps. If False, all time steps are used. Defaults to False.
            nSamples (int, optional): Number of samples for SHAP values computation. If False, all samples
                are used. Defaults to False.
            metric (str, optional): Metric for summarizing SHAP values ('mean', 'abs_mean', 'abs_mean_sign',
                'all', 'none'). Defaults to False.

        Returns:
            ndarray or tuple: Computed Shapley values based on the metric.
        """
        print(f"\nGetting Shapley values for variable {var}.\n")

        input_list = get_var_list(self.setup, self.setup.spcam_inputs)
        self.inputs = sorted(
            [Variable_Lev_Metadata.parse_var_name(p) for p in input_list],
            key=lambda x: self.setup.input_order_list.index(x),
        )
        self.input_vars_dict = ModelDiagnostics._build_vars_dict(self.inputs)

        self.output = Variable_Lev_Metadata.parse_var_name(var)
        self.output_vars_dict = ModelDiagnostics._build_vars_dict([self.output])

        # Train data as background
        model, inputs = self.models[var]

        self.train_gen = build_train_generator(
            self.input_vars_dict,
            self.output_vars_dict,
            self.setup,
            diagnostic_mode=True
        )
        with self.train_gen as train_gen:
            if itime == 'range':
                if not nTime:
                    data = h5py.File(train_gen.data_fn, "r")
                    X_train = self._normalize(data, train_gen)
                else:
                    n_batches = int((self.ngeo / self.setup.batch_size) * nTime)
                    X_train = np.zeros([self.ngeo * nTime, len(self.inputs)])

                    sIdx = 0
                    eIdx = self.setup.batch_size

                    for i in range(n_batches):
                        X_train[sIdx:eIdx, :] = train_gen[i][0]
                        sIdx += self.setup.batch_size
                        eIdx += self.setup.batch_size

        self.valid_gen = build_valid_generator(self.input_vars_dict, self.output_vars_dict, self.setup, test=True,
                                               diagnostic_mode=True)
        with self.valid_gen as valid_gen:
            if itime == 'range':
                if not nTime:
                    data = h5py.File(valid_gen.data_fn, "r")
                    X_test = self._normalize(data, valid_gen)
                else:
                    n_batches = nTime
                    X_test = np.zeros([self.ngeo * nTime, len(self.inputs)])

                    sIdx = 0
                    eIdx = self.ngeo

                    for i in range(n_batches):
                        X_test[sIdx:eIdx, :] = valid_gen[i][0]
                        sIdx += self.ngeo
                        eIdx += self.ngeo

        if not nSamples: nSamples = len(X_train)

        # Explain predictions of the model
        background = X_train[np.random.choice(X_train.shape[0], nSamples, replace=False)]
        test = X_test[np.random.choice(X_test.shape[0], nSamples, replace=False)]

        del X_test
        del X_train
        gc.collect()

        e = shap.DeepExplainer(model, background[:, inputs])
        shap_values = e.shap_values(test[:, inputs], check_additivity=False)

        shap_values_sign = np.mean(shap_values[0], axis=0)
        shap_values_sign[shap_values_sign < 0] = -1.
        shap_values_sign[shap_values_sign > 0] = 1.
        shap_values_mean = np.mean(shap_values[0], axis=0)
        shap_values_abs_mean = np.mean(np.absolute(shap_values[0]), axis=0)
        shap_values_abs_mean_sign = shap_values_abs_mean * shap_values_sign

        if not metric or metric == 'mean':
            return shap_values_mean, inputs, self.input_vars_dict
        elif metric == 'abs_mean':
            return shap_values_abs_mean, inputs, self.input_vars_dict
        elif metric == 'abs_mean_sign':
            return shap_values_abs_mean_sign, inputs, self.input_vars_dict
        elif metric == 'all':
            return shap_values_mean, shap_values_abs_mean, shap_values_abs_mean_sign, inputs, self.input_vars_dict
        elif metric == 'none':
            return e.expected_value.numpy(), test, shap_values, inputs, self.input_vars_dict
        else:
            print(f"metric not available, only: mean, abs_mean, abs_mean_sign and all; stop")
            exit()

    def _normalize(self, data, generator):
        data_x = data["vars"][:, generator.input_idxs]

        # Normalize
        data_x = generator.input_transform.transform(data_x)

        # Delete data to save memory
        del data
        gc.collect()
        return data_x

    # Plotting functions
    def plot_double_xy(
            self,
            itime,
            var,
            nTime=False,
            save=False,
            diff=False,
            stats=False,
            show_plot=True,
            **kwargs
    ):
        """
        Plots a cross-section (latitude-longitude) map of truth and prediction for a given variable.

        Args:
            itime (int or str): Time step or 'mean' for averaged data.
            var (str): Output variable.
            nTime (bool, optional): Number of time steps. Defaults to False.
            save (bool or str, optional): Directory to save the plot. Defaults to False.
                If save is a string, the plots and the data generating the plots are saved under that path.
            diff (bool, optional): Whether to plot the difference between truth and prediction. Defaults to False.
            stats (bool or list, optional): Statistics to include in the plot. Defaults to False.
            show_plot (bool, optional): Whether to display the plot. Defaults to True.

        Returns:
            None or tuple: Plot figure and axes if show_plot is True.
        """
        varname = var.var.value
        print(f"\nPlotting double_xy for variable {varname}\n")

        if stats is not False:
            t, p = self.get_truth_pred('range', var, nTime=nTime)
            psum, tsum, psqsum, tsqsum, sse = self.calc_sums(p, t, nTime)
            pmean, tmean, bias, mse, pred_mean, true_mean, \
                pred_sqmean, true_sqmean, pred_var, true_var, r2 = \
                self.calc_mean_stats(psum, tsum, psqsum, tsqsum, sse, nTime)
            local_vars = locals()
            if isinstance(stats, list):
                stats = [(s, local_vars[s]) for s in stats]
            else:
                stats = (stats, local_vars[stats])

            t, p = np.mean(t, axis=0), np.mean(p, axis=0)
        else:
            t, p = self.get_truth_pred(itime, var, nTime=nTime)

        if not isinstance(stats, list):
            fig, axes = self._plot_slices(t, p, itime, var=var, stype='xy', save=save, diff=diff,
                                          stats=[False, stats][stats is not False], nTime=nTime, **kwargs)
            if show_plot:
                return fig, axes
            else:
                plt.close(fig)
                print(f"\nClosed plot for variable {var}\n")
                return
        else:
            if show_plot:
                print(f"\nDisplay plot is not enabled when creating multiple plots.\n"
                      f"Continuing with creating and saving plots.\n")
            # Without stats, without and with difference
            fig, axes = self._plot_slices(t, p, itime, var=var, stype='xy', save=save, diff=False,
                                          stats=False, nTime=nTime, **kwargs)
            plt.close(fig)
            print(f"Closed plot for variable {var} for {itime} without diff.\n")
            fig, axes = self._plot_slices(t, p, itime, var=var, stype='xy', save=save, diff=True,
                                          stats=False, nTime=nTime, **kwargs)
            plt.close(fig)
            print(f"Closed plot for variable {var} for {itime} with diff.\n")

            for s in stats:
                fig, axes = self._plot_slices(t, p, itime, var=var, stype='xy', save=save, diff=False,
                                              stats=[False, s][s is not False], nTime=nTime, **kwargs)
                plt.close(fig)
                print(f"Closed plot for variable {var} with stats={s[0]}, diff=False.\n")
                fig, axes = self._plot_slices(t, p, itime, var=var, stype='xy', save=save, diff=True,
                                              stats=[False, s][s is not False], nTime=nTime, **kwargs)
                plt.close(fig)
                print(f"Closed plot for variable {var} with stats={s[0]}, diff=True.\n")

            del fig
            del axes
            gc.collect()
            return

    def plot_double_yz(
            self,
            var,
            varkeys,
            itime=1,
            nTime=False,
            ilon=1,
            save=False,
            diff=False,
            stats=False,
            show_plot=True,
            **kwargs
    ):
        """
        Plots a cross-section (latitude-pressure) for multiple vertical levels.

        Args:
            var (str): Output variable.
            varkeys (list): List of keys for vertical levels.
            itime (int, optional): Time step. Defaults to 1.
            nTime (bool, optional): Number of time steps. Defaults to False.
            ilon (int, optional): Longitude index. Defaults to 1.
            save (bool or str, optional): Directory to save the plot. Defaults to False.
                If save is a string, the plots and the data generating the plots are saved under that path.
            diff (bool, optional): Whether to plot the difference between truth and prediction. Defaults to False.
            stats (bool or list, optional): Statistics to include in the plot. Defaults to False.
            show_plot (bool, optional): Whether to display the plot. Defaults to True.

        Returns:
            None or tuple: Plot figure and axes if show_plot is True.
        """
        varname = var.var.value
        print(f"\nPlotting double_yz for variable {varname}.\n")

        # Allocate array
        truth = np.zeros([self.nlev, self.nlat])
        pred = np.zeros([self.nlev, self.nlat])

        if isinstance(stats, list):
            mean_stats = list()
            for i in range(len(stats)):
                mean_stats.append(np.zeros([self.nlev, self.nlat]))
        else:
            mean_stats = np.zeros([self.nlev, self.nlat])

        for var in varkeys:
            print(f"\nProcessing variable {var}.")
            iLev = ModelDiagnostics._build_vars_dict([var])[var.var.value.upper()][0]

            if stats is not False:
                t, p = self.get_truth_pred('range', var, nTime=nTime)
                psum, tsum, psqsum, tsqsum, sse = self.calc_sums(p, t, nTime)
                pmean, tmean, bias, mse, pred_mean, true_mean, \
                    pred_sqmean, true_sqmean, pred_var, true_var, r2_tmp = \
                    self.calc_mean_stats(psum, tsum, psqsum, tsqsum, sse, nTime)
                t, p = np.mean(t, axis=0), np.mean(p, axis=0)
            else:
                t, p = self.get_truth_pred(itime, var, nTime=nTime)

            if isinstance(ilon, int):
                truth[iLev, :] = t[:, ilon]
                pred[iLev, :] = p[:, ilon]
                if stats is not False:
                    hor_tsqmean = true_sqmean[:, ilon]
                    hor_tmean = true_mean[:, ilon]
                    hor_psqmean = pred_sqmean[:, ilon]
                    hor_pmean = pred_mean[:, ilon]
                    hor_mse = mse[:, ilon]
                    hor_rmse = np.sqrt(mse[:, ilon])

            elif ilon == 'mean':
                truth[iLev, :] = np.mean(t, axis=1)
                pred[iLev, :] = np.mean(p, axis=1)
                if stats is not False:
                    hor_tsqmean = np.mean(true_sqmean, axis=1)
                    hor_tmean = np.mean(true_mean, axis=1)
                    hor_psqmean = np.mean(pred_sqmean, axis=1)
                    hor_pmean = np.mean(pred_mean, axis=1)
                    hor_mse = np.mean(mse, axis=1)
                    hor_rmse = np.sqrt(np.mean(mse, axis=1))

            if stats is not False:
                hor_tvar = hor_tsqmean - hor_tmean ** 2
                hor_pvar = hor_psqmean - hor_pmean ** 2
                hor_r2 = 1 - (hor_mse / hor_tvar)

                local_vars = locals()
                if isinstance(stats, list):
                    for s, m in zip(stats, mean_stats):
                        m[iLev, :] = local_vars['hor_' + s]
                else:
                    mean_stats[iLev, :] = local_vars['hor_' + stats]

        if not isinstance(stats, list):
            fig, axes = self._plot_slices(truth, pred, itime, var=var, stype='yz', save=save, diff=diff,
                                          stats=[False, (stats, mean_stats)][stats is not False], **kwargs)
            if show_plot:
                return fig, axes
            else:
                plt.close(fig)
                print(f"\nClosed plot for variable {var}\n")
                return

        else:
            if show_plot:
                print(f"\nDisplay plot is not enabled when creating multiple plots.\n"
                      f"Continuing with creating and saving plots.\n")
            # Without stats, without and with difference
            fig, axes = self._plot_slices(truth, pred, itime, var=var, stype='yz', save=save, diff=False,
                                          stats=False, **kwargs)
            plt.close(fig)
            print(f"Closed plot for variable {var} for {itime} without diff.\n")

            fig, axes = self._plot_slices(truth, pred, itime, var=var, stype='yz', save=save, diff=True,
                                          stats=False, **kwargs)
            plt.close(fig)
            print(f"Closed plot for variable {var} for {itime} without diff.\n")

            for s, m in zip(stats, mean_stats):
                fig, axes = self._plot_slices(truth, pred, itime, var=var, stype='yz', save=save, diff=False,
                                              stats=[False, (s, m)][s is not False], **kwargs)
                plt.close(fig)
                print(f"Closed plot for variable {var} with stats={s}, diff=False.\n")
                fig, axes = self._plot_slices(truth, pred, itime, var=var, stype='yz', save=save, diff=True,
                                              stats=[False, (s, m)][s is not False], **kwargs)
                plt.close(fig)
                print(f"Closed plot for variable {var} with stats={s[0]}, diff=True.\n")

            del fig
            del axes
            gc.collect()
            return

    def _plot_slices(
            self,
            t,
            pred,
            itime,
            title='',
            unit='',
            var='',
            stype=False,
            save=False,
            diff=False,
            stats=False,
            nTime=False,
            **kwargs
    ):
        """Helper function to plot multiple slices of truth and predictions (with or without difference)"""
        varname = var.var.value
        n_slices = [3, 2][diff is False]
        n_slices = [n_slices + 1, n_slices][stats is False]
        fig, axes = plt.subplots(1, n_slices, figsize=(12, 5))

        extend = 'both'
        if 'vmin' in kwargs.keys() and 'vmax' in kwargs.keys():
            vmin, vmax = kwargs['vmin'], kwargs['vmax']
            kwargs.pop('vmin', None);
            kwargs.pop('vmax', None)
        else:
            vmin = np.min([np.min(pred), np.min(t)])
            vmax = np.max([np.max(pred), np.max(t)])
            if varname in ['tphystnd', 'phq']:
                vlim = np.max([np.abs(vmin), np.abs(vmax)]) / 2.
                vmin = -vlim;
                vmax = vlim
            elif varname in ['fsns', 'fsnt', 'flns', 'flnt', 'prect']:
                vmin = 0
                extend = 'max'

        if 'cmap' in kwargs.keys():
            cmap = kwargs['cmap'];
            kwargs.pop('cmap', None)
        else:
            cmap = cThemes[varname]
        cmap_diff = 'coolwarm'

        vars_to_plot = [np.array([pred, t]), np.array([pred, t, pred - t])][diff is True]
        labs_to_plot = [['Prediction', 'SPCAM', 'Prediction - SPCAM'],
                        ['Prediction', 'SPCAM']][diff is False]
        if stats is not False:
            vars_to_plot = np.insert(vars_to_plot, len(vars_to_plot), stats[1], axis=0)
            labs_to_plot.append(stats[0])
        for iSlice in range(n_slices):
            var_to_plot = vars_to_plot[iSlice]
            lab_to_plot = labs_to_plot[iSlice]
            if lab_to_plot == 'Prediction - SPCAM':
                vmin, vmax = -vmax / 1.2, vmax / 1.2
                cmap = cmap_diff
                extend = 'both'
            elif lab_to_plot == 'r2':
                vmin = 0.;
                vmax = 1.
                extend = 'min'
                cmap = 'Spectral_r'
            elif 'var' in lab_to_plot:
                vmin = 0;
                vmax = 3.5e-8
                cmap = 'Spectral_r'
                extend = 'max'
            elif lab_to_plot == 'mse':
                vmin = 0;
                vmax = np.max(stats[1])
                # vmin=0; vmax=1.e-4 # tphystnd
                # vmin=0; vmax=1.5e-7  # phq
                cmap = 'magma_r'
                extend = 'max'

            I = axes[iSlice].imshow(var_to_plot, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
            cb = fig.colorbar(I, ax=axes[iSlice], orientation='horizontal', extend=extend)
            cb.set_label(unit)
            axes[iSlice].set_title(lab_to_plot)
            if stype == 'xy':
                lat_ticks = [int(l) for l in range(len(self.latitudes)) if l % 9 == 0]
                lat_labels = [str(int(l)) for i, l in enumerate(self.latitudes) if i % 9 == 0]
                axes[iSlice].set_yticks(lat_ticks)
                axes[iSlice].set_yticklabels(lat_labels)
                axes[iSlice].set_ylabel('Latitude')
                lon_ticks = [int(l) for l in range(len(self.longitudes)) if l % 9 == 0]
                lon_labels = [str(int(l)) for i, l in enumerate(self.longitudes) if i % 9 == 0]
                axes[iSlice].set_xticks(lon_ticks)
                axes[iSlice].set_xticklabels(lon_labels)
                axes[iSlice].set_xlabel('Longitude')
            elif stype == 'yz':
                P_ticks = [int(press) for press in range(len(P)) if press % 5 == 0]
                P_label = [str(int(press)) for i, press in enumerate(P) if i % 5 == 0]
                axes[iSlice].set_yticks(P_ticks)
                axes[iSlice].set_yticklabels(P_label)
                axes[iSlice].set_ylabel('hPa')
                lat_ticks = [int(l) for l in range(len(self.latitudes)) if l % 9 == 0]
                lat_labels = [str(int(l)) for i, l in enumerate(self.latitudes) if i % 9 == 0]
                axes[iSlice].set_xticks(lat_ticks)
                axes[iSlice].set_xticklabels(lat_labels)
                axes[iSlice].set_xlabel('Latitude')

        fig.suptitle(title)
        plt.tight_layout()

        def _get_save_dir():
            if type(itime) is int:
                idx_time_str = f"step-{itime}"
            elif type(itime) is str:
                if nTime:
                    idx_time_str = f"{itime}-{nTime}"
                else:
                    idx_time_str = f"{itime}-all"
            else:
                raise ValueError(f"Unkown value for idx_time: {itime}")

            stats_str = f"_stats-{stats[0]}" if stats else ""
            diff_str = "_with_diff" if diff else "_no_diff"
            return idx_time_str + stats_str + diff_str

        if save:
            save_dir = os.path.join(save, _get_save_dir())
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            level = "-{}".format(var.level) if var.level else ""
            diff = "_diff" if diff else ""
            stats_str = f"_stats-{stats[0]}" if stats else ""
            t_steps = "-{}steps".format(nTime) if nTime else ""

            save_str = f"{varname}" + level + f"_map_time-{itime}" + t_steps + diff + stats_str + ".png"
            save_path = Path(save_dir, save_str)
            fig.savefig(save_path)

            print(f"\nSaved plot {save_path.name}.")

            # save prediction and truth
            pred_file = f"{varname}_cross_section_pred.p"
            with open(os.path.join(save_dir, pred_file), "wb") as f:
                pickle.dump(pred, f)
            print(f"\nSaved cross section prediction {pred_file}.")

            truth_file = f"{varname}_cross_section_truth.p"
            with open(os.path.join(save_dir, truth_file), "wb") as f:
                pickle.dump(t, f)
            print(f"\nSaved cross section truth {truth_file}.")

            # save stats
            if stats:
                stats_file = f"{varname}_cross_section_stats-{stats[0]}.p"
                with open(os.path.join(save_dir, stats_file), 'wb') as f:
                    pickle.dump(stats[1], f)
                print(f"\nSaved cross section stats {stats_file}.")

        return fig, axes

    def plot_double_profile(
            self,
            var,
            varkeys,
            itime=1,
            nTime=False,
            lats=[-90, 90],
            lons=[0., 359.],
            save=False,
            stats=False,
            show_plot=True,
            unit="",
            **kwargs
    ):
        """
        Plots vertical profiles of truth and predictions for a specified variable.

        Args:
            var (str): Output variable.
            varkeys (list): List of variable keys for vertical levels.
            itime (int, optional): Time step. Defaults to 1.
            nTime (bool, optional): Number of time steps. Defaults to False.
            lats (list, optional): Latitude range. Defaults to [-90, 90].
            lons (list, optional): Longitude range. Defaults to [0., 359.].
            save (bool or str, optional): Directory to save the plot. Defaults to False.
                If save is a string, the plots and the data generating the plots are saved under that path.
            stats (bool or str, optional): Statistics to include. Defaults to False.
            show_plot (bool, optional): Whether to display the plot. Defaults to True.
            unit (str, optional): Unit for the variable. Defaults to "".

        Returns:
            None or figure: Plot figure if show_plot is True.
        """
        varname = var.var.value
        print(f"\nPlotting double_profiles for variable {varname}\n")

        if not nTime and isinstance(itime, int):
            nTime = 1
        else:
            t, p = self.get_truth_pred(itime, varkeys[0], nTime=nTime)
            nTime = len(t)

        idx_lats = [find_closest_value(self.latitudes, lat) for lat in lats]
        idx_lons = [find_closest_longitude(self.longitudes, lon) for lon in lons]

        # Allocate array
        truth = np.zeros([nTime, self.nlev])
        pred = np.zeros([nTime, self.nlev])

        if isinstance(stats, list):
            mean_stats = list()
            for i in range(len(stats)):
                mean_stats.append(np.zeros([self.nlev, self.nlat]))
        else:
            mean_stats = np.zeros([self.nlev, self.nlat])

        for var in varkeys:
            print(f"\nProcessing variable {var}.")
            iLev = ModelDiagnostics._build_vars_dict([var])[var.var.value.upper()][0]
            t, p = self.get_truth_pred(itime, var, nTime=nTime)

            if itime == "range":
                truth[:, iLev] = np.average(
                    np.mean(t[:, idx_lats[0]:idx_lats[-1] + 1, idx_lons[0]:idx_lons[-1] + 1], axis=2),
                    weights=self.lat_weights[idx_lats[0]:idx_lats[-1] + 1], axis=1
                )
                pred[:, iLev] = np.average(
                    np.mean(p[:, idx_lats[0]:idx_lats[-1] + 1, idx_lons[0]:idx_lons[-1] + 1], axis=2),
                    weights=self.lat_weights[idx_lats[0]:idx_lats[-1] + 1], axis=1
                )
            elif itime == "mean" or isinstance(itime, int):
                truth[:, iLev] = np.average(
                    np.mean(t[idx_lats[0]:idx_lats[-1] + 1, idx_lons[0]:idx_lons[-1] + 1], axis=1),
                    weights=self.lat_weights[idx_lats[0]:idx_lats[-1] + 1], axis=0
                )
                pred[:, iLev] = np.average(
                    np.mean(p[idx_lats[0]:idx_lats[-1] + 1, idx_lons[0]:idx_lons[-1] + 1], axis=1),
                    weights=self.lat_weights[idx_lats[0]:idx_lats[-1] + 1], axis=0
                )

            if stats is not False:
                psum, tsum, psqsum, tsqsum, sse = self.calc_sums(p, t, nTime)
                pmean, tmean, bias, mse, pred_mean, true_mean, \
                    pred_sqmean, true_sqmean, pred_var, true_var, r2_tmp = \
                    self.calc_mean_stats(psum, tsum, psqsum, tsqsum, sse, nTime)
                hor_tsqmean = np.average(
                    np.mean(true_sqmean[idx_lats[0]:idx_lats[-1] + 1, idx_lons[0]:idx_lons[-1] + 1], axis=1),
                    weights=self.lat_weights[idx_lats[0]:idx_lats[-1] + 1]
                )
                hor_tmean = np.average(
                    np.mean(true_mean[idx_lats[0]:idx_lats[-1] + 1, idx_lons[0]:idx_lons[-1] + 1], axis=1),
                    weights=self.lat_weights[idx_lats[0]:idx_lats[-1] + 1]
                )
                hor_mse = np.average(
                    np.mean(mse[idx_lats[0]:idx_lats[-1] + 1, idx_lons[0]:idx_lons[-1] + 1], axis=1),
                    weights=self.lat_weights[idx_lats[0]:idx_lats[-1] + 1])
                hor_tvar = hor_tsqmean - hor_tmean ** 2
                hor_r2 = 1 - (hor_mse / hor_tvar)
                local_vars = locals()
                if isinstance(stats, list):
                    for s, m in zip(stats, mean_stats):
                        m[iLev, :] = local_vars['hor_' + s]
                else:
                    mean_stats[iLev, :] = local_vars['hor_' + stats]

        if not isinstance(stats, list):
            fig = self._plot_profiles(truth, pred, itime, varname=varname, nTime=nTime, lats=lats, lons=lons,
                                      save=save, stats=[False, (stats, mean_stats)][stats is not False], unit=unit,
                                      **kwargs)
            if show_plot:
                return fig
            else:
                plt.close(fig)
                print(f"\nClosed plot for variable {var}\n")
                return
        else:
            if show_plot:
                print(f"\nDisplay plot is not enabled when creating multiple plots.\n"
                      f"Continuing with creating and saving plots.\n")
            # Without stats, without and with difference
            fig = self._plot_profiles(truth, pred, itime, varname=varname, nTime=nTime, lats=lats, lons=lons,
                                      save=save, stats=False, unit=unit, **kwargs)

            plt.close(fig)
            print(f"Closed plot for variable {var} for {itime}.\n")

            for s, m in zip(stats, mean_stats):
                fig = self._plot_profiles(truth, pred, itime, varname=varname, nTime=nTime, lats=lats, lons=lons,
                                          save=save, stats=[False, (s, m)][s is not False], unit=unit, **kwargs)
                plt.close(fig)
                print(f"Closed plot for variable {var} with stats={s}.\n")

            del fig
            gc.collect()
            return

    def _plot_profiles(
            self,
            t,
            p,
            itime,
            title='',
            unit='',
            varname='',
            nTime=False,
            lats=[-90, 90],
            lons=[0., 359.],
            save=False,
            stats=False,
            **kwargs
    ):
        """Helper function to plot vertical profiles of truth and predictions"""

        n_cols = [1, 2][stats is not False]
        fig = plt.figure(1, figsize=(12, 5))
        ax1 = plt.subplot(1, n_cols, 1)

        vmin = np.min([np.min(p), np.min(t)])
        vmax = np.max([np.max(p), np.max(t)])
        vlim = np.max([np.abs(vmin), np.abs(vmax)]) / 2.
        vmin = -vlim;
        vmax = vlim

        vars_to_plot = [p, t]
        labs_to_plot = ['Prediction', 'SPCAM']
        if stats is not False:
            labs_to_plot.append(stats[0])

        colors = ['b', 'b', 'k']
        linestyles = ['--', '-', '-']
        for iVar in range(len(vars_to_plot)):
            var_to_plot = vars_to_plot[iVar]
            lab_to_plot = labs_to_plot[iVar]
            ax1.plot(
                np.mean(var_to_plot, axis=0),
                P,
                label=lab_to_plot,
                color=colors[iVar],
                linestyle=linestyles[iVar],
                alpha=.8,
                **kwargs
            )
        ax1.invert_yaxis()
        ax1.set_xlim(vmin, vmax)
        ax1.set_xlabel(f"{varname} ({unit})")
        ax1.set_ylabel('Pressure (hPa)')
        ax1.legend(loc=0)

        if stats is not False:
            ax2 = plt.subplot(1, n_cols, 2)
            ax2.plot(
                stats[1],
                P,
                'k-',
                label=varname,
                alpha=.8,
                **kwargs
            )
            ax2.invert_yaxis()
            ax2.set_xlabel(stats[0])
            ax2.set_ylabel('Pressure (hPa)')
            handles, labels = ax2.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax2.legend(by_label.values(), by_label.keys(), loc=0)
            # ax2.legend(loc=0)

        fig.suptitle(title)

        def _get_save_dir():
            if type(itime) is int:
                idx_time_str = f"step-{itime}"
            elif type(itime) is str:
                if nTime:
                    idx_time_str = f"{itime}-{nTime}"
                else:
                    idx_time_str = f"{itime}-all"
            else:
                raise ValueError(f"Unknown value for idx_time: {itime}")

            lats_str = f"_lats{lats[0]}_{lats[1]}" if lats else ""
            lons_str = f"_lons{lons[0]}_{lons[1]}" if lons else ""
            stats_str = f"_stats-{stats[0]}" if stats else ""

            return idx_time_str + lats_str + lons_str + stats_str

        if save:
            save_dir = os.path.join(save, _get_save_dir())
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            t_steps = f"-{nTime}steps" if nTime else ""
            stats_str = f"_stats-{stats[0]}" if stats else ""
            save_str = f"{varname}" + f"_profile_time-{itime}" + t_steps + stats_str + ".png"
            save_path = Path(save_dir, save_str)
            fig.savefig(save_path)
            print(f"\nSaved plot {save_path.name}.")

            # save prediction and truth
            pred_file = f"{varname}_profile_pred.p"
            with open(os.path.join(save_dir, pred_file), "wb") as f:
                pickle.dump(p, f)

            truth_file = f"{varname}_profile_truth.p"
            with open(os.path.join(save_dir, truth_file), "wb") as f:
                pickle.dump(t, f)

            # save stats
            if stats:
                stats_file = f"{varname}_stats-{stats[0]}.p"
                with open(os.path.join(save_dir, stats_file), 'wb') as f:
                    pickle.dump(stats[1], f)
                print(f"\nSaved stats {stats_file}.")

        return fig

    def calc_sums(self, p, t, nTime):
        """Computes the sum, square sum, and error for predictions and truth data.

        Args:
            p (ndarray): Predictions.
            t (ndarray): Truth data.
            nTime (int): Number of time steps.

        Returns:
            tuple: Sums, square sums, and sum of squared errors (SSE).
        """
        # Allocate stats arrays
        psum = np.zeros((self.nlat, self.nlon))
        tsum = np.copy(psum)
        sse = np.copy(psum)
        psqsum = np.copy(psum)
        tsqsum = np.copy(psum)
        #        for it in tqdm(range(nTime)):
        for it in range(nTime):
            # Compute statistics
            psum += p[it]
            tsum += t[it]
            psqsum += p[it] ** 2
            tsqsum += t[it] ** 2
            sse += (t[it] - p[it]) ** 2
        return psum, tsum, psqsum, tsqsum, sse

    def calc_mean_stats(self, psum, tsum, psqsum, tsqsum, sse, nTime):
        """
        Calculates mean statistics including bias, MSE, and R^2.

        Args:
            psum (ndarray): Sum of predictions.
            tsum (ndarray): Sum of truth values.
            psqsum (ndarray): Square sum of predictions.
            tsqsum (ndarray): Square sum of truth values.
            sse (ndarray): Sum of squared errors.
            nTime (int): Number of time steps.

        Returns:
            tuple: Various statistics including mean, variance, and R^2.
        """
        pmean = psum / nTime
        tmean = tsum / nTime
        bias = pmean - tmean
        mse = sse / nTime
        pred_mean = psum / nTime
        true_mean = tsum / nTime
        pred_sqmean = psqsum / nTime
        true_sqmean = tsqsum / nTime
        pred_var = psqsum / nTime - pmean ** 2
        true_var = tsqsum / nTime - tmean ** 2
        r2 = 1. - (mse / true_var)
        r2 = ma.masked_invalid(r2)
        return pmean, tmean, bias, mse, pred_mean, true_mean, \
            pred_sqmean, true_sqmean, pred_var, true_var, r2

    # Statistics computation
    def compute_stats(self, itime, var, nTime=False):
        """
        Computes various statistics for the specified variable at a given time step.
        Statistics are saved in the attribute `self.stats`.

        Args:
            itime (int or str): Time step or 'mean' for averaged data.
            var (str): Output variable.
            nTime (bool, optional): Number of time steps. Defaults to False.

        Returns:
            None
        """
        print(f"\nComputing horizontal stats for variable {var}\n")

        t, p = self.get_truth_pred(itime, var, nTime=nTime)
        nTime = len(t) if nTime is False else nTime  # Time steps

        psum, tsum, psqsum, tsqsum, sse = self.calc_sums(p, t, nTime)

        # Compute average statistics
        pmean, tmean, bias, mse, pred_mean, true_mean, \
            pred_sqmean, true_sqmean, pred_var, true_var, r2 = \
            self.calc_mean_stats(psum, tsum, psqsum, tsqsum, sse, nTime)

        # Compute horizontal stats: single value per var
        self.stats = {}
        self.stats['hor_tsqmean'] = np.average(np.mean(true_sqmean, axis=1), weights=self.lat_weights)
        self.stats['hor_tmean'] = np.average(np.mean(true_mean, axis=1), weights=self.lat_weights)
        self.stats['hor_mse'] = np.average(np.mean(mse, axis=1), weights=self.lat_weights)
        self.stats['hor_tvar'] = self.stats['hor_tsqmean'] - self.stats['hor_tmean'] ** 2
        self.stats['hor_r2'] = 1 - (self.stats['hor_mse'] / self.stats['hor_tvar'])

    def mean_stats(self):
        """
        Computes the mean statistics for each output variable and returns a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the mean statistics for each variable.
        """
        df = pd.DataFrame(index=self.valid_gen.output_vars,
                          columns=list(self.stats.keys()))
        for ivar, var in enumerate(self.valid_gen.output_vars):
            for stat_name, stat in self.stats.items():
                # Stats have shape [lat, lon]
                if 'hor_' in stat_name:
                    df.loc[var, stat_name] = stat
                else:
                    df.loc[var, stat_name] = ma.mean(stat[..., self.get_output_var_idx(var)])
        self.stats_df = df
        return df

    def get_path(self):
        """Generates a path based on model metadata for saving or loading models.

        Returns:
            Path: Path object representing the generated file path.
        """
        base_path = self.setup.nn_output_path
        path = Path(base_path, self.setup.model_type)
        if self.setup.model_type == "CausalSingleNN" or self.setup.model_type == "CorrSingleNN":
            if self.setup.area_weighted:
                cfg_str = "a{pc_alpha}-t{threshold}-latwts/"
            else:
                cfg_str = "a{pc_alpha}-t{threshold}/"
            path = path / Path(
                cfg_str.format(pc_alpha=self.setup.pc_alpha, threshold=self.setup.threshold)
            )
        str_hl = str(self.setup.hidden_layers).replace(", ", "_")
        str_hl = str_hl.replace("[", "").replace("]", "")
        path = path / Path(
            "hl_{hidden_layers}-act_{activation}-e_{epochs}/".format(
                hidden_layers=str_hl,
                activation=self.setup.activation,
                epochs=self.setup.epochs,
            )
        )
        return path

    @staticmethod
    def _build_vars_dict(list_variables):
        """
        Converts a list of Variable_Lev_Metadata into a dictionary used by the data generator.

        Args:
            list_variables (list): List of Variable_Lev_Metadata.

        Returns:
            dict: Dictionary mapping dataset names to their respective levels or None for 2D variables.
        """
        vars_dict = dict()
        for variable in list_variables:
            ds_name = variable.var.ds_name  # Name used in the dataset
            if variable.var.dimensions == 2:
                ds_name = 'LHF_nsDELQ' if ds_name == 'LHF_NSDELQ' else ds_name
                vars_dict[ds_name] = None
            elif variable.var.dimensions == 3:
                levels = vars_dict.get(ds_name, list())
                levels.append(variable.level_idx)
                vars_dict[ds_name] = levels
        return vars_dict
