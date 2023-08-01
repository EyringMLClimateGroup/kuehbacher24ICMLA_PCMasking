from utils.constants import ANCIL_FILE
from utils.variable import Variable_Lev_Metadata
from utils.utils import read_ancilaries, find_closest_value, find_closest_longitude, get_weights  # , get_pressure
from pathlib import Path
from neural_networks.load_models import get_var_list
from neural_networks.data_generator import build_train_generator, build_valid_generator
from neural_networks.cbrain.utils import load_pickle
from neural_networks.cbrain.cam_constants import *
import numpy as np
from math import pi
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ipykernel.kernelapp import IPKernelApp


def in_notebook():
    return IPKernelApp.initialized()


if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
import shap

cThemes = {'tphystnd': 'coolwarm',
           'phq': 'RdBu',
           'fsns': 'Reds',
           'flns': 'Reds',
           'fsnt': 'Reds',
           'flnt': 'Reds',
           'prect': 'PuBu'}


class ModelDiagnostics():
    def __init__(self, setup, models, nlat=64, nlon=128, nlev=30, ntime=48):

        self.nlat, self.nlon, self.nlev = nlat, nlon, nlev
        self.ngeo = nlat * nlon
        self.setup = setup
        self.models = models

        self.levels, self.latitudes, self.longitudes = read_ancilaries(
            Path(ANCIL_FILE)
        )
        self.lat_weights = np.cos(self.latitudes * pi / 180.)

    def reshape_ngeo(self, x, nTime=False):
        #        return x.reshape(self.nlat, self.nlon, -1)
        if nTime:
            return x.reshape(nTime, self.nlat, self.nlon)
        else:
            return x.reshape(self.nlat, self.nlon)

    def get_output_var_idx(self, var):
        var_idxs = self.valid_gen.norm_ds.var_names[self.valid_gen.output_idxs]
        var_idxs = np.where(var_idxs == var)[0]
        return var_idxs

    def get_truth_pred(self, itime, var, nTime=False):
        input_list = get_var_list(self.setup, self.setup.spcam_inputs)
        self.inputs = sorted(
            [Variable_Lev_Metadata.parse_var_name(p) for p in input_list],
            key=lambda x: self.setup.input_order_list.index(x),
        )
        self.input_vars_dict = ModelDiagnostics._build_vars_dict(self.inputs)

        # self.save_dir=False
        self.input_pca_vars_dict = False
        if self.setup.do_pca_nn:
            self.inputs_pca = self.inputs[:int(self.setup.n_components)]
            self.input_pca_vars_dict = ModelDiagnostics._build_vars_dict(self.inputs_pca)

        self.output = Variable_Lev_Metadata.parse_var_name(var)
        self.output_vars_dict = ModelDiagnostics._build_vars_dict([self.output])

        self.valid_gen = build_valid_generator(
            self.input_vars_dict,
            self.output_vars_dict,
            self.setup,
            test=True,
            # save_dir=self.save_dir, 
            input_pca_vars_dict=self.input_pca_vars_dict,
            diagnostic_mode=True
        )
        with self.valid_gen as valid_gen:

            model, inputs = self.models[var]

            if isinstance(itime, int):
                X, truth = valid_gen[itime]
                pred = model.predict_on_batch(X[:, inputs])
                # For CASTLE, we just want the prediction for Y
                if self.setup.nn_type == "castleNN":
                    pred = pred[:, 0]

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
                    # For CASTLE, we just want the prediction for Y
                    if self.setup.nn_type == "castleNN":
                        p_tmp = p_tmp[:, 0]

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
                if not nTime: nTime = len(train_gen)

                # X_train, _ = train_gen[0] #TODO: Change this to retrieve all the data
                n_batches = int((self.ngeo / self.setup.batch_size) * nTime)
                # print(f"n_batches: {n_batches}")
                X_train = np.zeros([self.ngeo * nTime, len(self.inputs)])

                sIdx = 0
                eIdx = self.setup.batch_size

                for i in range(n_batches):
                    X_train[sIdx:eIdx, :] = train_gen[i][0]
                    sIdx += self.setup.batch_size
                    eIdx += self.setup.batch_size

        self.valid_gen = build_valid_generator(
            self.input_vars_dict,
            self.output_vars_dict,
            self.setup,
            test=True,
            diagnostic_mode=True
        )
        with self.valid_gen as valid_gen:
            if itime == 'range':
                # X_test, _ = valid_gen[0] #TODO: Change this to retrieve all the data
                if not nTime: nTime = len(valid_gen)
                n_batches = nTime
                X_test = np.zeros([self.ngeo * nTime, len(self.inputs)])

                sIdx = 0
                eIdx = self.ngeo
                for i in range(n_batches):
                    X_test[sIdx:eIdx, :] = valid_gen[i][0]
                    sIdx += self.ngeo
                    eIdx += self.ngeo

        if not nSamples: nSamples = len(X_train)

        # explain predictions of the model
        background = X_train[np.random.choice(X_train.shape[0], nSamples, replace=False)]
        test = X_test[np.random.choice(X_test.shape[0], nSamples, replace=False)]

        e = shap.DeepExplainer(model, background[:, inputs])
        shap_values = e.shap_values(test[:, inputs], check_additivity=False)

        shap_values_sign = np.mean(shap_values[0], axis=0)
        shap_values_sign[shap_values_sign < 0] = -1.
        shap_values_sign[shap_values_sign > 0] = 1.
        shap_values_mean = np.mean(shap_values[0], axis=0)
        shap_values_abs_mean = np.mean(np.absolute(shap_values[0]), axis=0)
        shap_values_abs_mean_sign = shap_values_abs_mean * shap_values_sign

        # shap_values_mean = valid_gen.output_transform.inverse_transform(np.mean(shap_values[0],axis=0))
        # print(f'transformed shap_values_mean: {valid_gen.output_transform.inverse_transform(shap_values_mean)}')

        # return model, X_train, X_test, self.input_vars_dict
        if not metric or metric == 'mean':
            return shap_values_mean, inputs, self.input_vars_dict
        elif metric == 'abs_mean':
            return shap_values_abs_mean, inputs, self.input_vars_dict
        elif metric == 'abs_mean_sign':
            return shap_values_abs_mean_sign, inputs, self.input_vars_dict
        else:
            print(f"metric not available, only: mean, abs_mean and abs_mean_sign; stop")
            exit()

    # Plotting functions
    def plot_double_xy(
            self,
            itime,  # can be int specifying time step or "mean"
            var,
            nTime=False,
            save=False,
            diff=False,
            stats=False,
            show_plot=True,
            **kwargs
    ):
        varname = var.var.value
        print(f"\nPlotting double_xy for variable {varname}\n")

        if stats is not False:
            t, p = self.get_truth_pred('range', var, nTime=nTime)
            psum, tsum, psqsum, tsqsum, sse = self.calc_sums(p, t, nTime)
            pmean, tmean, bias, mse, pred_mean, true_mean, \
                pred_sqmean, true_sqmean, pred_var, true_var, r2 = \
                self.calc_mean_stats(psum, tsum, psqsum, tsqsum, sse, nTime)
            stats = (stats, locals()[stats])
            t, p = np.mean(t, axis=0), np.mean(p, axis=0)
        else:
            t, p = self.get_truth_pred(itime, var, nTime=nTime)

        if show_plot:
            return self.plot_slices(
                t,
                p,
                itime,
                var=var,
                stype='xy',
                save=save,
                diff=diff,
                stats=[False, stats][stats is not False],
                nTime=nTime,
                **kwargs
            )
        else:
            fig, axes = self.plot_slices(
                t,
                p,
                itime,
                var=var,
                stype='xy',
                save=save,
                diff=diff,
                stats=[False, stats][stats is not False],
                **kwargs
            )
            plt.close(fig)
            print(f"\nClosed plot for variable {var}\n")
            return

    def plot_double_yz(
            self,
            var,  # this var is redundant
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
        varname = var.var.value
        print(f"\nPlotting double_yz for variable {varname}\n")

        # Allocate array
        truth = np.zeros([self.nlev, self.nlat])
        pred = np.zeros([self.nlev, self.nlat])
        mean_stats = np.zeros([self.nlev, self.nlat])
        for var in varkeys:
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
                mean_stats[iLev, :] = locals()['hor_' + stats]

        if show_plot:
            return self.plot_slices(truth, pred, itime, var=var, stype='yz', save=save, diff=diff,
                                    stats=[False, (stats, mean_stats)][stats is not False], **kwargs)
        else:
            fig, axes = self.plot_slices(truth, pred, itime, var=var, stype='yz', save=save, diff=diff,
                                         stats=[False, (stats, mean_stats)][stats is not False], **kwargs)
            plt.close(fig)
            print(f"\nClosed plot for variable {var}\n")
            return

    def plot_slices(
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
                #                 vmin=0; vmax=np.max(stats[1])
                vmin = 0;
                vmax = 3.5e-8
                cmap = 'Spectral_r'
                extend = 'max'
            elif lab_to_plot == 'mse':
                vmin = 0;
                vmax = np.max(stats[1])
                #                 vmin=0; vmax=1.e-4 # tphystnd
                # vmin=0; vmax=1.5e-7  # phq
                cmap = 'magma_r'
                extend = 'max'
            #             else:
            #                 lmin=vmin; lmax=vmax
            #                 cmap_theme=cmap
            # todo: this is where range fails because var_to_plot is 3d instead of 2d
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

        if save:
            Path(save).mkdir(parents=True, exist_ok=True)
            level = "-{}".format(var.level) if var.level else ""
            diff = "_diff" if diff else ""
            stats = "_stats" if stats else ""
            t_steps = "-{}steps".format(nTime) if nTime else ""
            save_str = "{varname}" + level + "_map_time-{itime}" + t_steps + diff + stats + ".png"
            save_path = Path(save, save_str.format(varname=varname, itime=itime))
            fig.savefig(save_path)
            print(f"Saved plot {save_path.name}.")
        return fig, axes

    def get_profiles(
            self,
            var,
            varkeys,
            itime=1,
            nTime=False,
            lats=[-90, 90],
            lons=[0., 359.],
            stats=False,
            **kwargs
    ):
        varname = var.var.value

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
        mean_stats = np.zeros([self.nlev])
        for var in varkeys:
            iLev = ModelDiagnostics._build_vars_dict([var])[var.var.value.upper()][0]
            t, p = self.get_truth_pred(itime, var, nTime=nTime)

            truth[:, iLev] = np.average(
                np.mean(t[:, idx_lats[0]:idx_lats[-1] + 1, idx_lons[0]:idx_lons[-1] + 1], axis=2),
                weights=self.lat_weights[idx_lats[0]:idx_lats[-1] + 1], axis=1
            )
            pred[:, iLev] = np.average(
                np.mean(p[:, idx_lats[0]:idx_lats[-1] + 1, idx_lons[0]:idx_lons[-1] + 1], axis=2),
                weights=self.lat_weights[idx_lats[0]:idx_lats[-1] + 1], axis=1
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
                mean_stats[iLev] = locals()['hor_' + stats]

        return truth, pred, mean_stats

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
        mean_stats = np.zeros([self.nlev])
        for var in varkeys:
            iLev = ModelDiagnostics._build_vars_dict([var])[var.var.value.upper()][0]
            t, p = self.get_truth_pred(itime, var, nTime=nTime)

            truth[:, iLev] = np.average(
                np.mean(t[idx_lats[0]:idx_lats[-1] + 1, idx_lons[0]:idx_lons[-1] + 1], axis=2),
                weights=self.lat_weights[idx_lats[0]:idx_lats[-1] + 1], axis=1
            )
            pred[:, iLev] = np.average(
                np.mean(p[idx_lats[0]:idx_lats[-1] + 1, idx_lons[0]:idx_lons[-1] + 1], axis=2),
                weights=self.lat_weights[idx_lats[0]:idx_lats[-1] + 1], axis=1
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
                mean_stats[iLev] = locals()['hor_' + stats]

        if show_plot:
            return self.plot_profiles(
                truth,
                pred,
                itime,
                varname=varname,
                nTime=nTime,
                lats=lats,
                lons=lons,
                save=save,
                stats=[False, (stats, mean_stats)][stats is not False],
                unit=unit,
                **kwargs
            )
        else:
            fig = self.plot_profiles(
                truth,
                pred,
                itime,
                varname=varname,
                nTime=nTime,
                lats=lats,
                lons=lons,
                save=save,
                stats=[False, (stats, mean_stats)][stats is not False],
                unit=unit,
                **kwargs
            )
            plt.close(fig)
            print(f"\nClosed plot for variable {var}\n")
            return

    def plot_profiles(
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
            ax2.legend(loc=0)

        fig.suptitle(title)
        if save:
            Path(save).mkdir(parents=True, exist_ok=True)
            save_path = Path(f"{save}/{varname}_profile_time-{itime}.png")
            fig.savefig(save_path)
            print(f"\nSaved profile plot {save_path.name}")

        return fig

    def calc_sums(self, p, t, nTime):
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
        """Compute statistics in for [lat, lon, var, lev]"""

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
        """Get average statistics for each variable and returns dataframe"""
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
        """ Generate a path based on this model metadata """
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
        elif self.setup.model_type == "pcaNN":
            if self.setup.area_weighted:
                cfg_str = "pcs{n_components}-latwts/"
            else:
                cfg_str = "pcs{n_components}/"
            path = path / Path(
                cfg_str.format(n_components=self.setup.n_components)
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
        """ Convert the given list of Variable_Lev_Metadata into a
        dictionary to be used on the data generator.
        
        Parameters
        ----------
        list_variables : list(Variable_Lev_Metadata)
            List of variables to be converted to the dictionary format
            used by the data generator
        
        Returns
        -------
        vars_dict : dict{str : list(int)}
            Dictionary of the form {ds_name : list of levels}, where
            "ds_name" is the name of the variable as stored in the
            dataset, and "list of levels" a list containing the indices
            of the levels of that variable to use, or None for 2D
            variables.
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


# Additional Plots (not included within the diagnostic class)
def plot_profiles(
        vars_dict,
        varname='',
        title='',
        unit='',
        save=False,
        stats=False,
        **kwargs
):
    n_cols = [1, 2][stats is not False]
    fig = plt.figure(1, figsize=(12, 5))
    ax1 = plt.subplot(1, n_cols, 1)
    if stats is not False:
        ax2 = plt.subplot(1, n_cols, 2)

    truth = False
    for i, iMod in enumerate(vars_dict.keys()):
        if not truth:
            ax1.plot(
                np.mean(vars_dict[iMod][varname]['t'], axis=0),
                P,
                color='k',
                label='SPCAM',
                alpha=.8,
                **kwargs
            )
            truth = True
        if iMod == 'SingleNN' or iMod == 'CausalNN':
            ax1.plot(
                np.mean(vars_dict[iMod][varname]['p'], axis=0),
                P,
                label=vars_dict[iMod][varname]['lab'],
                alpha=.8,
                **kwargs
            )
        if stats is not False:
            ax2.plot(
                vars_dict[iMod][varname][stats][1],
                P,
                label=vars_dict[iMod][varname]['lab'],
                alpha=.8,
                **kwargs
            )

    ax1.invert_yaxis()
    #    ax1.set_xlim(vmin, vmax)
    ax1.set_xlabel(f" {varname} ({unit})")
    ax1.set_ylabel('Pressure (hPa)')
    ax1.legend(loc=0)

    if stats is not False:
        ax2.invert_yaxis()
        if stats == 'r2':
            ax2.set_xlim(0., 1.)
        elif stats == 'mse':
            ax2.set_xlim(0., np.max(vars_dict[iMod][varname][stats][1]))
        ax2.set_xlabel(vars_dict[iMod][varname][stats][0])
        ax2.set_ylabel('Pressure (hPa)')
        ax2.legend(loc=0)

    fig.suptitle(title)
    if save is not False:
        Path(save).mkdir(parents=True, exist_ok=True)
        save_path = Path(f"{save}/{varname}_profile.png")
        fig.savefig(save_path)
        print(f"\nSaved profile plot {save_path.name}")

    plt.show()
