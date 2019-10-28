from lib2.Measurement import *
from lib2.VNATimeResolvedDispersiveMeasurement import *
from lib2.IQPulseSequence import *

from numpy.linalg import pinv # you will have to deal with this, FUCKING PSEUDOINVERSE, BABY
import numpy as np
from scipy.optimize import least_squares, curve_fit


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

    def in_bounds(self, p0, bounds):
        true_arr = []
        for i, val in enumerate(p0):
            if( val >= bounds[0][i] and val <= bounds[1][i] ):
                true_arr.append(True)
            else:
                true_arr.append(False)

        return true_arr


    def _fit_complex_curve(self, X, data):
        p0, bounds = self._generate_fit_arguments(X, data)
        try:
            p0, err = curve_fit(lambda x, *params: real(self._model(x, *params)) + imag(self._model(x, *params)),
                                X, real(data) + imag(data),
                                p0=p0,
                                bounds=bounds)
        except Exception as e:
            pass
            # this is silent due to the frequent exceptions thrown
            # print("VNATDM1D->_fit_complex_curve-> curve fit filed:", e)
            # print(p0,bounds)
            # raise e
        finally:
            try:
                result = least_squares(self._cost_function, p0, args=(X, data),
                                       bounds=bounds, x_scale="jac", max_nfev=10000, ftol=1e-5)
            except Exception as e:
                print("VNATDM1D->_fit_complex_curve->least_squares filed:", e)
                print(p0, bounds)
                raise e
            sigma = std(abs(self._model(X, *result.x) - data))

            if self._fit_params is not None:
                try:
                    result_2 = least_squares(self._cost_function, self._fit_params,
                                             args=(X, data), bounds=bounds, x_scale="jac",
                                             max_nfev=1000, ftol=1e-5)
                except Exception as e:
                    print("VNATDM1D->_fit_complex_curve->least squares (self._fit_params is not None) filed:", e)
                    print(p0, bounds)
                    raise e

                sigma_2 = std(abs(self._model(X, *result_2.x) - data))
                if sigma_2 < sigma:
                    result = result_2
                    sigma = sigma_2

            return result, sqrt(diag(sigma ** 2 * pinv(result.jac.T.dot(result.jac))))

    def fit(self, verbose=True):

        meas_data = self.get_data()
        # hotfix. KeyError happens sometimes. Due to the fact that
        # I manually set _data to {} in order overcome to avoid
        # fit_complex_curve "'x0' is infeasible" exception
        # this reset operation happens from the measurement thread
        # so, it is what it is
        if "data" not in meas_data.keys():
            return

        data = meas_data["data"][meas_data["data"] != 0]
        if len(data) < 5:
            return

        X = self._prepare_data_for_plot(meas_data)[0]
        X = X[:len(data)]

        try:
            result, err = self._fit_complex_curve(X, data)
            if result.success:
                self._fit_params = result.x
                self._fit_errors = err
        except Exception as e:
            print("Fit failed unexpectedly:", e)

    def _prepare_figure(self):
        fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
        fig.canvas.set_window_title(self._name)
        axes = ravel(axes)
        return fig, axes, (None, None)

    def _prepare_data_for_plot(self, data):
        return data[self._parameter_names[0]] / 1e3, data["data"]

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
        # axes["phase"].set_xlabel(xlabel)
        axes["imag"].set_xlabel(xlabel)
        plt.tight_layout(pad=2)
        self._plot_fit(axes)

    def _generate_annotation_string(self, opt_params, err):
        """
        Should be implemented in child classes
        """
        pass

    def _annotate_fit_plot(self, idx, ax, opt_params, err):
        h_pos = mean(ax.get_xlim())
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
            self.fit(verbose=False)
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