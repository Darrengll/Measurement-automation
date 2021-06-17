from matplotlib import pyplot as plt
from IPython.display import clear_output
from lib2.fulaut.qubit_spectra import *
from importlib import reload
import lib2.ResonatorDetector

reload(lib2.ResonatorDetector)
from lib2.ResonatorDetector import *
from loggingserver import LoggingServer
from lib2.ExperimentParameters import *

import scipy
from scipy import *
from scipy.optimize import *
from scipy.signal import *
import numpy as np


class AnticrossingOracle():
    """
    This class automatically processes anticrossing spectral data for
    different types of qubits and if_freq arrangements between the qubits and
    resonators
    """

    qubit_spectra = {"transmon": transmon_spectrum}

    def __init__(self, qubit_type, sts_result, plot=False,
                 fast_res_detect=True, hints={}, remove_outliers=True, silent=False):
        self._qubit_spectrum = AnticrossingOracle.qubit_spectra[qubit_type]
        self._sts_result = sts_result
        self._plot = plot
        self._logger = LoggingServer.getInstance("")
        self._fast = True
        self._fast_res_detect = fast_res_detect
        self._noisy_data = False
        self._remove_outliers = remove_outliers
        self._hints = hints
        self._res_points = []
        self._iteration_counter = 0
        self._silent = silent

        self._extract_data()

        fr_estimate = mean(self._res_points[:, 1])

        if "fqmax_below" in self._hints:
            if not self._hints["fqmax_below"]:
                q_freq_range = slice(fr_estimate, 12.1e9, 100e6)
            else:
                q_freq_range = slice(1, fr_estimate, 100e6)
        else:
            q_freq_range = slice(1e9, 12.1e9, 100e6)

        self._default_parameter_ranges = \
            {
                "fr": slice(fr_estimate - 1e6, fr_estimate + 1.1e6, 1e6),
                "g": slice(20e6, 40.1e6, 20e6 / 5),
                "period": None,  # always frozen in brute
                "sws_current": None,  # always frozen in brute
                "fqmax": q_freq_range,
                "d": slice(0.1, 0.81, .1)
            }
        self._frozen_parameters = {}

    def _generate_brute_grids(self):
        parameter_grids = []
        for idx, parameter_name in enumerate(self._default_parameter_ranges.keys()):
            if parameter_name not in self._frozen_parameters:
                parameter_grids.append(self._default_parameter_ranges[parameter_name])

        return parameter_grids

    def freeze_parameters(self, frozen_parameters_dict):
        self._frozen_parameters = frozen_parameters_dict

    def launch(self):

        self._period = self._find_period()
        potential_sweet_spots = self._find_potential_sweet_spots()

        args = (self._res_points[:, 0], self._res_points[:, 1])

        self._loss = 1e100
        # We are not sure where the sweet spot is, so let's choose the best
        # fit among two possibilities:
        for sweet_spot_cur in potential_sweet_spots:

            mean_cur = mean(self._res_points[:, 0])
            distance_to_sws = abs(mean_cur - sweet_spot_cur)
            shift = round(distance_to_sws / self._period) * self._period * sign(
                mean_cur - sweet_spot_cur)
            sweet_spot_cur += shift

            frozen_parameters = {key: self._hints[key] for key in self._default_parameter_ranges if key in self._hints}
            frozen_parameters["period"] = self._period
            frozen_parameters["sws_current"] = sweet_spot_cur
            self.freeze_parameters(frozen_parameters)
            brute_grids = self._generate_brute_grids()

            self._iteration_counter = 0
            brute_result = brute(self._cost_function,
                                 brute_grids,
                                 args=args,
                                 finish=None)
            brute_loss = \
                sqrt(self._cost_function(brute_result, *args) / len(self._res_points)) / 1e6

            brute_full_result = self._active_params_to_full_params(list(brute_result))

            frozen_parameters = {key: self._hints[key] for key in self._default_parameter_ranges if key in self._hints}
            self.freeze_parameters(frozen_parameters)
            self._iteration_counter = 0
            nm_result = minimize(self._cost_function,
                                 self._full_parameters_to_active_parameters(brute_full_result),
                                 args=args, method="Nelder-Mead").x
            loss = \
                sqrt(self._cost_function(nm_result, *args) / len(self._res_points)) / 1e6

            if loss < self._loss:
                self._brute_opt_params = brute_full_result
                self._brute_loss = brute_loss
                self._opt_params = self._active_params_to_full_params(nm_result)
                self._loss = loss

            if loss < 0.05:
                break

        res_freq, g, period, sweet_spot_cur, q_freq, d = self._opt_params

        if self._plot:
            plt.figure()
            plt.plot(self._res_points[:, 0], self._res_points[:, 1], '.',
                     label="Data")
            plt.plot([sweet_spot_cur], [mean(self._res_points[:, 1])], '+')

            # p0 = [mean((res_freq_2, res_freq_2)), 0.03, period,
            #                                         sweet_spot_cur, 10, 0.6]
            # plt.plot(self._curs, self._model(self._curs, p0), "o")
            plt.plot(self._res_points[:, 0],
                     self._model_fast(self._res_points[:, 0],
                                      self._brute_opt_params, False),
                     "yellow", ls=":", label="Brute")
            plt.plot(self._res_points[:, 0],
                     self._model_fast(self._res_points[:, 0],
                                      self._opt_params, False),
                     "orange", ls="-", marker=".", label="Final")
            plt.legend()
            plt.gcf().set_size_inches(15, 5)

        return self._opt_params, self._loss

    def _extract_data(self, plot=False):
        try:
            data = self._sts_result.get_data()
        except:
            # maybe we have raw dict
            data = self._sts_result
        param_keys = ["Current [A]", "bias", "Voltage [V]"]
        for param_key in param_keys:
            if param_key in data:
                param_values = data[param_key]

        try:
            freqs = data["frequency"]
        except:
            freqs = data["Frequency [Hz]"]

        self._data = data["data"]

        data = self._data

        # convert data to delay -- works for reflection AND trasmission
        # data = self._sts_result._remove_delay(freqs, data)
        unwrapped_phase = unwrap(angle(data), axis=1)
        filter_window = data.shape[1] // 20
        if filter_window % 2 == 0:
            filter_window += 1
        filter_polyorder = 3
        self._filtered_uphase = filtered_uphase = savgol_filter(unwrapped_phase, filter_window,
                                                                filter_polyorder, axis=1)
        self._delay = delay = abs(diff(self._filtered_uphase, axis=1))

        # Taking delay peaks higher than half of the distance between the median
        # level and the highest point for all 2D data
        threshold = (delay.max() -
                     0.75 * (delay.max() - median(delay)))

        res_points = []
        self._extracted_indices = []
        self._extraction_types = []

        for idx, row in enumerate(self._delay):
            extrema_inhdices = argrelextrema(row, greater, order=10)[0]
            extrema_inhdices = extrema_inhdices[row[extrema_inhdices] > threshold]

            if len(extrema_inhdices) > 0:
                RD = ResonatorDetector(freqs, data[idx], plot=False,
                                       fast=self._fast_res_detect,
                                       type=GlobalParameters().resonator_type)
                result = RD.detect()
                if result is not None:
                    res_points.append((param_values[idx], result[0]))
                    self._extraction_types.append("fit")
                    self._extracted_indices.append(idx)
                else:
                    highest_extremum_idx = \
                        extrema_inhdices[argmax(row[extrema_inhdices])]
                    res_points.append((param_values[idx], freqs[highest_extremum_idx]))
                    self._extraction_types.append("max_delay")
                    self._extracted_indices.append(idx)

        self._res_points = array(res_points)
        self._extraction_types = array(self._extraction_types)
        self._freqs = freqs
        self._curs = param_values

        if self._remove_outliers:
            self._remove_outlier_points()

        if self._plot:
            plt.figure()
            plt.plot(self._res_points[self._extraction_types == "fit", 0],
                     self._res_points[self._extraction_types == "fit", 1], 'C1.',
                     label="Extracted points (fit)")
            plt.plot(self._res_points[self._extraction_types == "max_delay", 0],
                     self._res_points[self._extraction_types == "max_delay", 1], 'C3.',
                     label="Extracted points (max. delay)")
            plt.pcolormesh(param_values, freqs[:-1], delay.T)
            plt.legend()
            plt.gcf().set_size_inches(15, 5)

    def _remove_outlier_points(self):
        median_diff = median(abs(diff(self._res_points[:, 1])))
        #     print(median_diff)
        to_drop = []
        for idx in range(1, len(self._res_points) - 1):
            point_prev, point, point_next = self._res_points[idx - 1:idx + 2]
            diff_prev = point[1] - point_prev[1]
            diff_next = point_next[1] - point[1]
            diff_diff = abs(abs(diff_next) - abs(diff_prev))
            #         print("%.2f %.2f %.2f"%(diff_next/median_diff, diff_prev/median_diff, diff_diff/median_diff), end="\n")
            if abs(diff_next) > 15 * median_diff and abs(diff_prev) > 15 * median_diff:
                if diff_diff < 5 * median_diff:
                    to_drop.append(idx)
        self._extraction_types = np.delete(self._extraction_types, to_drop, axis=0)
        self._res_points = np.delete(self._res_points, to_drop, axis=0)
        self._extracted_indices = np.delete(self._extracted_indices, to_drop, axis=0)

    def _find_period(self):
        extracted_no_mean = self._res_points[:, 1] - mean(self._res_points[:, 1])
        extracted_zero_padded = scipy.zeros(len(self._curs))
        extracted_zero_padded[self._extracted_indices] = extracted_no_mean
        data = extracted_zero_padded

        self._corr = corr = correlate(data, data, "full")[data.size - 1:]
        peaks = argrelextrema(corr, greater, order=10)[0]
        try:
            period = peaks[argmax(corr[peaks])]
            # print(peaks, period)
            return self._curs[period] - self._curs[0]
        except ValueError:
            return 1.5 * ptp(self._curs)

    def _model_square(self, duty, phase, x):
        return square(2 * pi * x / self._period - phase, duty)

    def _cost_function_sweet_spots(self, p, x, y):
        duty, phase = p
        fit_data = self._model_square(duty, phase, x)
        return -sum(fit_data * y)

    def _find_potential_sweet_spots(self):
        data = self._res_points[:, 1] - mean(self._res_points[:, 1])
        duty, phase = brute(self._cost_function_sweet_spots,
                            ((0, 1), (-pi, pi)),
                            Ns=50,
                            args=(self._res_points[:, 0], data),
                            full_output=0)
        self._duty, self._phase = duty, phase
        sws1 = phase / 2 / pi * self._period + self._period * duty / 2
        sws2 = phase / 2 / pi * self._period - self._period * (1 - duty) / 2

        if max(abs(diff(data))) > 0.5 * ptp(data):
            # we probably have anticrossings
            max_num_of_anticrossings = ceil(ptp(self._curs) / self._period) * 2
            min_number_of_anticrossings = floor(ptp(self._curs) / self._period) * 2
            large_derivatives = where(abs(diff(data)) > 0.5 * ptp(data))[0]

            if min_number_of_anticrossings <= len(large_derivatives) <= max_num_of_anticrossings:
                # everything is fine, not noise
                return [sws2]

        elif max(abs(diff(data))) < 0.1 * ptp(data):
            # we probably have smooth curves
            return [sws1]

        # we will check both, noisy scan
        return sws1, sws2

    @staticmethod
    def _eigenlevels(f_q, f_r, g0):
        E0 = (f_r - f_q) / 2
        g = g0 * sqrt(f_q) / sqrt(6e9)
        E1 = f_r - 1 / 2 * sqrt(4 * g ** 2 + (f_q - f_r) ** 2)
        E2 = f_r + 1 / 2 * sqrt(4 * g ** 2 + (f_q - f_r) ** 2)
        return array([E0 - E0, E1 - E0, E2 - E0])

    def _model_fast(self, curs, params, plot=False):
        f_r, g = params[:2]
        qubit_params = params[2:]
        #     phis_fine = linspace(phis[0],phis[-1], 1000)
        f_qs = self._qubit_spectrum(curs, *qubit_params)

        freq_span = self._freqs[-1] - self._freqs[0]
        levels = self._eigenlevels(f_qs, f_r, g)

        if plot:
            plt.plot(curs, levels[0, :])
            plt.plot(curs, levels[1, :])
            plt.plot(curs, levels[2, :])
            plt.ylim(self._freqs[0], self._freqs[-1])

        upper_limit = f_r + freq_span
        lower_limit = f_r - freq_span

        res_freqs_model = zeros_like(curs) + 0.5 * (self._freqs[-1] + self._freqs[0])
        idcs1 = where(logical_and(lower_limit < levels[1, :],
                                  levels[1, :] < upper_limit))
        idcs2 = where(logical_and(lower_limit < levels[2, :],
                                  levels[2, :] < upper_limit))

        res_freqs_model[idcs1] = levels[1, :][idcs1]
        res_freqs_model[idcs2] = levels[2, :][idcs2]

        return res_freqs_model

    def _active_params_to_full_params(self, active_params):
        params = []
        counter = 0
        for param_name in self._default_parameter_ranges:
            if param_name in self._frozen_parameters:
                params.append(self._frozen_parameters[param_name])
            else:
                params.append(active_params[counter])
                counter += 1
        return params

    def _full_parameters_to_active_parameters(self, full_parameters):
        active_params = []
        for idx, param_name in enumerate(self._default_parameter_ranges):
            if param_name not in self._frozen_parameters:
                active_params.append(full_parameters[idx])
        return active_params

    def _cost_function(self, active_params, curs, res_freqs):

        params = self._active_params_to_full_params(active_params)
        cost = (self._model_fast(curs, params) - res_freqs) ** 2

        if self._iteration_counter % 100 == 0 and not self._silent:
            clear_output(wait=True)
            print(self._iteration_counter,
                  (("{:.4e}, " * len(params))[:-2]).format(*params),
                  "loss:",
                  "%.2f" % (sqrt(sum(cost) / len(curs)) / 1e6), "MHz")
        self._iteration_counter += 1
        return sum(cost)

    def get_res_points(self):
        return self._res_points
