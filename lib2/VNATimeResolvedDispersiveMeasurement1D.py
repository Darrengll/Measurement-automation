from lib2.VNATimeResolvedDispersiveMeasurement import \
    VNATimeResolvedDispersiveMeasurement, VNATimeResolvedDispersiveMeasurementResult
# from lib2.IQPulseSequence import *

import numpy as np
import scipy as sp
# cannot access 'sp.optimize' unless you import it excplicitly
import scipy.optimize
import matplotlib.pyplot as plt

import inspect

class VNATimeResolvedDispersiveMeasurement1D(VNATimeResolvedDispersiveMeasurement):

    def __init__(self, name, sample_name, devs_aliases_map,
                 plot_update_interval=1):
        super().__init__(name, sample_name, devs_aliases_map, plot_update_interval=plot_update_interval)

    def set_fixed_parameters(self, pulse_sequence_parameters, detect_resonator=True, plot_resonator_fit=True,
                             **dev_params):
        """
        :param dev_params:
            Minimum expected keys and elements expected in each:
                'vna'
                'q_awg': 0
                'ro_awg'
        """
        dev_params['vna'][0]["power"] = dev_params['ro_awg'][0]["calibration"] \
            .get_radiation_parameters()["lo_power"]

        super().set_fixed_parameters(pulse_sequence_parameters,
                                     detect_resonator=detect_resonator,
                                     plot_resonator_fit=plot_resonator_fit,
                                     **dev_params)

    def set_swept_parameters(self, par_name, par_values):
        swept_pars = {par_name: (self._output_pulse_sequence, par_values)}
        super().set_swept_parameters(**swept_pars)

    def _output_pulse_sequence(self, sequence_parameter):
        self._pulse_sequence_parameters[self._swept_parameter_name] = sequence_parameter
        super()._output_pulse_sequence()


class VNATimeResolvedDispersiveMeasurement1DResult( \
        VNATimeResolvedDispersiveMeasurementResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._x_axis_units = "$\mu$s"
        self._annotation_bbox_props = dict(boxstyle="round", fc="white",
                                           ec="black", lw=1, alpha=0.5)
        self._annotation_v_pos = "bottom"
        self._data_formats_used = ["real", "imag"]
        self._data_points_marker_size = 7
        self._lines = [None] * 2
        self._fit_lines = [None] * 2
        self._anno = [None] * 2

    def _cost_function(self, params, x, data):
        return np.abs(self._model(x, *params) - data)

    @staticmethod
    def in_bounds(p0, bounds):
        true_arr = []
        for i, val in enumerate(p0):
            if( val >= bounds[0][i] and val <= bounds[1][i] ):
                true_arr.append(True)
            else:
                true_arr.append(False)

        return true_arr

    def _fit_complex_curve(self, X, data):
        p0, bounds = self._generate_fit_arguments(X, data)

        # fit with initial fit arguments or with previous fit result
        try:
            result_1 = sp.optimize.least_squares(
                self._cost_function, p0,
                args=(X, data),
                bounds=bounds, x_scale="jac",
                max_nfev=10000, ftol=1e-5
            )
            # calculating standard deviations for fit parameters
            residuals = self._cost_function(result_1.x, X, data)
            hessian = (result_1.jac.T).dot(result_1.jac)
            if len(residuals) > len(p0):
                reduced_chi2_1 = np.sum(np.square(residuals))/(len(residuals) -
                                                    len(p0))
            else:
                reduced_chi2_1 = np.inf

            err_1 = np.sqrt(
                np.diag(reduced_chi2_1 * np.linalg.pinv(hessian))
            )

            # the return values. Can be substituted in the next fit
            result = result_1
            err = err_1
        except Exception as e:
            print("VNATimeResolvedDispersiveMeasurement1D"
                  "->_fit_complex_curve->least_squares failed:", e)
            print("p0 and bounds:")
            print(p0, bounds)
            print("out of bounds values indexes:")
            for i, (lb, hb, p) in enumerate(zip(*(list(bounds) + [p0]))):
                if p > hb or p < lb:
                    print(f"{i}: {lb} <= {p} <= {hb}")
            raise e

        # try fit with initial conditions equal to previous fit iteration
        # result
        if self._fit_params is not None:
            try:
                result_2 = sp.optimize.least_squares(
                    self._cost_function, self._fit_params,
                    args=(X, data),
                    bounds=bounds,
                    x_scale="jac",
                    max_nfev=1000, ftol=1e-5
                )
                # calculating standard deviations for fit parameters
                residuals = self._cost_function(result_1.x, X, data)
                hessian = (result_2.jac.T).dot(result_2.jac)
                if (len(residuals) > len(p0)):
                    reduced_chi2_2 = np.sum(residuals ** 2) / (len(residuals) -
                                                               len(p0))
                else:
                    reduced_chi2_2 = np.inf
                err_2 = np.sqrt(
                    np.diag(reduced_chi2_2 * np.linalg.pinv(hessian))
                )

                # if this fit is better than the previous one
                # we may substitute return value to be the values of
                # this fit
                if (result_2.cost < result_1.cost):
                    result = result_2
                    err = err_2
            except Exception as e:
                print("VNATimeResolvedDispersiveMeasurement1D"
                      "->_fit_complex_curve->least squares "
                      "(self._fit_params is not None) failed:", e)
                print(p0, bounds)
                raise e

        return result, err

    def fit(self):

        meas_data = self.get_data()
        # hotfix. KeyError happens sometimes. Due to the fact that
        # I manually set _data to {} in order overcome to avoid
        # fit_complex_curve "'x0' is infeasible" exception
        # this reset operation happens from the measurement thread
        # so, it is what it is
        if "data" not in meas_data.keys():
            return

        data = meas_data["data"][:self._iter_idx_ready[0]+1]  # 1D array
        if len(data) < (len(inspect.signature(self._model).parameters) - 1):
            return

        X = self._prepare_data_for_plot(meas_data)[0]
        X = X[:len(data)]

        try:
            result, err = self._fit_complex_curve(X, data)
            if result.success:
                self._fit_params = result.x
                self._fit_errors = err
        except Exception as e:
            print("VNATimeResolvedDispersiveMeasurement1D->fit: "
                  "fit failed unexpectedly:", e)

    def _prepare_figure(self):
        fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
        fig.canvas.set_window_title(self._name)
        axes = np.ravel(axes)
        return fig, axes, (None, None)

    def _prepare_data_for_plot(self, data):
        return data[self._parameter_names[0]], data["data"]

    def _plot(self, data):
        axes = self._axes
        axes = dict(zip(self._data_formats_used, axes))
        if "data" not in data.keys():
            return

        X, Y_raw = self._prepare_data_for_plot(data)

        for idx, name in enumerate(self._data_formats_used):
            Y = self._data_formats[name][0](Y_raw)
            Y = Y[Y != 0]
            ax = axes[name]
            if self._lines[idx] is None or not self._dynamic:
                ax.clear()
                ax.grid()
                ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
                self._lines[idx], = ax.plot(X[:len(Y)], Y, "C%d" % idx, ls=":", marker="o",
                                            markerfacecolor='none',
                                            markersize=self._data_points_marker_size)
                ax.set_xlim(X[0], X[-1])
                ax.set_ylabel(self._data_formats[name][1])
            else:
                self._lines[idx].set_xdata(X[:len(Y)])
                self._lines[idx].set_ydata(Y)
                ax.relim()
                ax.autoscale_view()

        xlabel = self._parameter_names[0][0].upper() + \
                 self._parameter_names[0][1:].replace("_", " ") + \
                 " [%s]" % self._x_axis_units

        axes["imag"].set_xlabel(xlabel)
        plt.tight_layout(pad=2)
        self._plot_fit(axes)

    def _generate_annotation_string(self, opt_params, err):
        """
        Should be implemented in child classes
        """
        pass

    def _annotate_fit_plot(self, idx, ax, opt_params, err):
        h_pos = np.mean(ax.get_xlim())
        v_pos = .9 * ax.get_ylim()[0] + .1 * ax.get_ylim()[1] \
            if self._annotation_v_pos == "bottom" else \
            .1 * ax.get_ylim()[0] + .9 * ax.get_ylim()[1]
        annotation_string = self._generate_annotation_string(opt_params, err)
        if self._anno[idx] is None or not self._dynamic:
            self._anno[idx] = ax.annotate(annotation_string, (h_pos, v_pos),
                                          bbox=self._annotation_bbox_props, ha="center")
        else:
            self._anno[idx].remove()
            self._anno[idx] = ax.annotate(annotation_string, (h_pos, v_pos),
                                          bbox=self._annotation_bbox_props, ha="center")
            # print(h_pos, v_pos)
            # print(self._anno[idx])

    def _plot_fit(self, axes, do_fit=True):
        if do_fit:
            self.fit()
        if self._fit_params is None:
            return

        for idx, name in enumerate(self._data_formats_used):
            ax = axes[name]
            opt_params = self._fit_params
            err = self._fit_errors
            data = self.get_data()
            # hotfix. KeyError happens sometimes. Due to the fact that
            # I manually set _data to {} in order to avoid
            # fit_complex_curve "'x0' is infeasible" exception
            # this reset operation happens from the measurement thread
            # so, it is what it is
            if "data" not in data.keys():
                return

            X = self._prepare_data_for_plot(data)[0]
            Y = self._data_formats[name][0](self._model(X, *opt_params))
            if self._fit_lines[idx] is None or not self._dynamic:
                self._fit_lines[idx], = ax.plot(X, Y, "C%d" % idx)
            else:
                self._fit_lines[idx].set_xdata(X)
                self._fit_lines[idx].set_ydata(Y)
            self._annotate_fit_plot(idx, ax, opt_params, err)
            plt.draw()

    def __getstate__(self):
        d = super().__getstate__()
        d['_lines'] = [None]*2
        d['_fit_lines'] = [None]*2
        d['_anno'] = [None]*2
        return d