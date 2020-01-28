from matplotlib import pyplot as plt
from IPython.display import clear_output
from lib2.fulaut.qubit_spectra import *
from importlib import reload
import lib2.ResonatorDetector
reload(lib2.ResonatorDetector)
from lib2.ResonatorDetector import *
from lib2.LoggingServer import *
from lib2.GlobalParameters import *

import scipy
from scipy import *
from scipy.optimize import *
from scipy.signal import *
import numpy as np

class AnticrossingOracle():

    """
    This class automatically processes anticrossing spectral data for
    different types of qubits and frequency arrangements between the qubits and
    resonators
    """

    qubit_spectra = {"transmon":transmon_spectrum}

    def __init__(self, qubit_type, sts_result, plot=False,
                 fast_res_detect = True, hints = [], remove_outliers = True, silent = False):
        self._qubit_spectrum = AnticrossingOracle.qubit_spectra[qubit_type]
        self._sts_result = sts_result
        self._plot = plot
        self._minimum_points_between_zeroes = 5
        self._minimum_points_around_zero = 5
        self._distance_from_intersection = 2
        self._logger = LoggingServer.getInstance()
        self._fast = True
        self._fast_res_detect = fast_res_detect
        self._noisy_data = False
        self._remove_outliers = remove_outliers
        self._hints = hints
        self._res_points = []
        self._iteration_counter = 0
        self._extract_data()
        self._silent = silent

    def launch(self):

        self._period = self._find_period()
        potential_sweet_spots = self._find_potential_sweet_spots()

        d_range = slice(0.1, 0.81, 0.9/9)
        mean_freq = mean(self._res_points[:, 1])
        f_range = slice(mean_freq-1e6, mean_freq+1.1e6, 1e6)
        if "fqmax_above" in self._hints:
            q_freq_range = slice(mean_freq, 12.1e9, 100e6)
        elif "fqmax_below" in self._hints:
            q_freq_range = slice(4, mean_freq, 100e6)
        else:
            q_freq_range = slice(4e9, 12.1e9, 100e6)

        g_range = slice(20e6, 40.1e6, 20e6/5)
        Ns = 3
        args = (self._res_points[:, 0], self._res_points[:, 1])
        # self._logger.debug(str(res_freq)

        best_fit_loss = 1e100
        best_fitresult = None
        # We are not sure where the sweet spot is, so let's choose the best
        # fit among two possibilities:
        for sweet_spot_cur in potential_sweet_spots:

            mean_cur = mean(self._res_points[:, 0])
            distance_to_sws = abs(mean_cur - sweet_spot_cur)
            shift = round(distance_to_sws / self._period) * self._period * sign(
                mean_cur - sweet_spot_cur)
            sweet_spot_cur += shift

            self._iteration_counter = 0
            result = brute(self._brute_cost_function,
                           (f_range, g_range, q_freq_range, d_range), Ns=Ns,
                           args=args+(self._period, sweet_spot_cur), finish=None)

            freq, g, q_max_freq, d = list(result)
            full_params = [freq, g, self._period, sweet_spot_cur, q_max_freq, d]

            self._iteration_counter = 0
            result = minimize(self._cost_function, full_params,
                              args=args, method="Nelder-Mead")

            loss =\
                sqrt(self._cost_function(result.x, *args)/len(self._res_points))/1e6

            brute_loss =\
                sqrt(self._cost_function(full_params, *args)/len(self._res_points))/1e6

            if loss < best_fit_loss:
                self._brute_opt_params = full_params
                self._brute_loss = brute_loss
                self._opt_params = result.x
                self._loss = loss
                best_fit_loss = loss
                best_fitresult = result

            if loss<0.05:
                break

        res_freq, g, period, sweet_spot_cur, q_freq, fd = best_fitresult.x

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
                     best_fitresult.x, False),
                     "orange", ls="-", marker=".", label="Final")
            plt.legend()
            plt.gcf().set_size_inches(15, 5)

        best_fitresult.x[0] = best_fitresult.x[0]
        best_fitresult.x[4] = best_fitresult.x[4]
        return best_fitresult.x, best_fit_loss

    def _extract_data(self, plot=False):
        try:
            data = self._sts_result.get_data()
        except:
            # maybe we have raw dict
            data = self._sts_result
        param_keys = ["Current [A]", "current", "Voltage [V]"]
        for param_key in param_keys:
            if param_key in data:
                param_values = data[param_key]

        try:
            freqs = data["frequency"]
        except:
            freqs = data["Frequency [Hz]"]

        self._data = data["data"]


        data = self._data

        #convert data to delay -- works for reflection AND trasmission
        #data = self._sts_result._remove_delay(freqs, data)
        unwrapped_phase = unwrap(angle(data), axis=1)
        filter_window = data.shape[1] // 10
        if filter_window % 2 == 0:
            filter_window += 1
        filter_polyorder = 3
        self._filtered_uphase = filtered_uphase = savgol_filter(unwrapped_phase,
                                                                filter_window,
                                                                filter_polyorder,
                                                                axis=1)
        self._delay = delay = abs(diff(self._filtered_uphase, axis = 1))

        res_freqs = []

        # def comlex_ptp_estimation(Z):
        #     point0 = Z[0, 0]
        #     point1 = Z.ravel()[argmax(abs(Z-point0))]
        #     point2 = Z.ravel()[argmax(abs(Z-point1))]
        #     point3 = Z.ravel()[argmax(abs(Z-point2))]
        #     return abs(point3-point2)

        # mean_derivative = mean(abs(diff(filtered_uphase)))
        # data_ptp = ptp(filtered_uphase)

        # if mean_derivative > 0.2*data_ptp:
        #     # we probably have a lot of noise
        #     self._noisy_data = True

        # preprocessed_data = filtered_data if self._noisy_data else data
        # self._preprocessed_data = preprocessed_data = data

        # Taking delay peaks higher than half of the distance between the median
        # transmission level and the highest point for all 2D data
        threshold = (delay.max() - \
                    0.5*(delay.max() - median(delay)))/100

        res_points = []
        self._extracted_indices = []
        self._extraction_types = []

        for idx, row in enumerate(self._delay):
            extrema_inhdices = argrelextrema(row, less, order=10)[0]
            extrema_inhdices = extrema_inhdices[row[extrema_inhdices] > threshold]

            if len(extrema_inhdices) > 0:
                if GlobalParameters().resonator_types['reflection'] == True:
                    RD = ResonatorDetector(freqs, data[idx], plot=False,
                                       fast=self._fast_res_detect, type='reflection')
                else:
                    RD = ResonatorDetector(freqs, data[idx], plot=False,
                                       fast=self._fast_res_detect, type='transmission')
                result = RD.detect()
                if result is not None:
                    res_points.append((param_values[idx], result[0]))
                    self._extraction_types.append("fit")
                    self._extracted_indices.append(idx)
                else:
                    highest_extremum_idx =\
                        extrema_inhdices[argmax(row[extrema_inhdices])]
                    res_points.append((param_values[idx], freqs[highest_extremum_idx]))
                    self._extraction_types.append("max_delay")
                    self._extracted_indices.append(idx)

        self._res_points = array(res_points)
        self._freqs = freqs
        self._curs = param_values

        if self._remove_outliers:
            self._remove_outlier_points()

        if self._plot:
            plt.figure()
            plt.plot(self._res_points[:,0], self._res_points[:,1], 'C1.',
                        label="Extracted points")
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
        self._res_points = np.delete(self._res_points, to_drop, axis=0)
        self._extracted_indices = np.delete(self._extracted_indices, to_drop, axis=0)

    def _find_period(self):
        extracted_no_mean = self._res_points[:,1]-mean(self._res_points[:,1])
        extracted_zero_padded = scipy.zeros(len(self._curs))
        extracted_zero_padded[self._extracted_indices] = extracted_no_mean
        data = extracted_zero_padded

        self._corr = corr = correlate(data, data, "full")[data.size-1:]
        peaks = argrelextrema(corr, greater, order=10)[0]
        try:
            period = peaks[argmax(corr[peaks])]
            # print(peaks, period)
            return self._curs[period] - self._curs[0]
        except ValueError:
            return 1.5 * ptp(self._curs)


    def _model_square(self, duty, phase, x):
        return square(2*pi*x/self._period-phase, duty)

    def _cost_function_sweet_spots(self, p, x, y):
        duty, phase = p
        fit_data = self._model_square(duty, phase, x)
        return -sum(fit_data*y)

    def _find_potential_sweet_spots(self):
        data = self._res_points[:,1]-mean(self._res_points[:,1])
        duty, phase = brute(self._cost_function_sweet_spots,
                            ((0, 1), (-pi, pi)),
                            Ns = 50,
                            args=(self._res_points[:,0], data),
                            full_output=0)
        self._duty, self._phase = duty, phase
        sws1 = phase/2/pi*self._period + self._period*duty/2
        sws2 = phase/2/pi*self._period - self._period*(1-duty)/2

        if max(abs(diff(data))) > 0.5*ptp(data):
            # we probably have anticrossings
            max_num_of_anticrossings = ceil(ptp(self._curs)/self._period)*2
            min_number_of_anticrossings = floor(ptp(self._curs)/self._period)*2
            large_derivatives = where(abs(diff(data)) > 0.5*ptp(data))[0]

            if min_number_of_anticrossings <= len(large_derivatives) <= max_num_of_anticrossings:
                # everything is fine, not noise
                return [sws2]

        elif max(abs(diff(data))) < 0.1*ptp(data):
            # we probably have smooth curves
            return [sws1]

        # we will check both, noisy scan
        return sws1, sws2


    # def find_resonator_intersections(self):
    #     plt.plot(self._res_points[:,0],
    #              ones_like(self._res_points[:,1])*mean(self._res_points[:,1]),
    #              'C2', label="Mean")
    #     data = (mean(self._res_points[:,1])-self._res_points[:,1])
    #
    #     raw_intersections = where((data[:-1]*data[1:])<0)[0]+1
    #
    #     refined_intersections = []
    #     current_intersection = raw_intersections[0]
    #     refined_intersections.append(current_intersection)
    #
    #     for raw_intersection in raw_intersections[1:]:
    #         distance = raw_intersection - current_intersection
    #         if distance > self._minimum_points_between_zeroes:
    #             current_intersection = raw_intersection
    #             refined_intersections.append(current_intersection)
    #
    #     fine_intersections = []
    #
    #     points_around_zero = self._minimum_points_around_zero
    #     distance_from_intersection = self._distance_from_intersection
    #     for intersection in refined_intersections:
    #         if intersection <= distance_from_intersection\
    #                     or  intersection>=(len(data)-distance_from_intersection):
    #             continue
    #
    #         if points_around_zero-1 >= intersection:
    #             left_edge = 0
    #             right_edge = intersection+points_around_zero
    #
    #         elif points_around_zero-1 < \
    #                 intersection<\
    #                     (len(data)-points_around_zero):
    #             left_edge = intersection-points_around_zero
    #             right_edge = intersection+points_around_zero
    #         else:
    #             right_edge = len(data)
    #             left_edge = intersection-points_around_zero
    #
    #         left_points = data[left_edge:intersection-distance_from_intersection]
    #         right_points = data[intersection+distance_from_intersection:right_edge]
    #
    #         self._logger.debug("Intersection: "+str(intersection)\
    #                             +", points: "+str((left_points, right_points)))
    #
    #         if len(set(left_points<0))==1\
    #             and len(set(right_points<0))==1\
    #             and left_points[0]*right_points[0]<0:
    #             fine_intersections.append(intersection)
    #
    #     self._logger.debug("Fine intersections: "+str(fine_intersections))
    #     return fine_intersections

    # @staticmethod
    # def _transmission(f_q, f_probe, f_r, g):
    #     κ = 1e6
    #     γ = 1e6
    #     return abs(1/(κ+1j*(f_r-f_probe)+(g**2)/(γ+1j*(f_q-f_probe))))**2

    @staticmethod
    def _eigenlevels(f_q, f_r, g):
        E0 = (f_r - f_q)/2
        E1 = f_r-1/2*sqrt(4*g**2+(f_q-f_r)**2)
        E2 = f_r+1/2*sqrt(4*g**2+(f_q-f_r)**2)
        return array([E0-E0, E1-E0, E2-E0])


    # def _model(self, curs, params, plot_colours = False, freqs_fine_number=5e3):
    #
    #     f_r, g = params[:2]
    #     qubit_params = params[2:]
    # #     phis_fine = linspace(phis[0],phis[-1], 1000)
    #     f_qs = self._qubit_spectrum(curs, *qubit_params)
    #
    #     span = ptp(self._freqs)
    #     freqs_fine = linspace(self._freqs[0]-span*0.5,
    #                             self._freqs[-1]+span*0.5,
    #                                 freqs_fine_number)
    #
    #     XX, YY = meshgrid(f_qs, freqs_fine)
    #     transmissions = self._transmission(XX, YY, f_r, g)
    #
    #     if plot_colours:
    #         plt.pcolormesh(curs, freqs_fine, transmissions)
    #         plt.colorbar()
    #
    #     res_freqs_model = []
    #     for row in transmissions.T:
    #         res_freqs_model.append(freqs_fine[argmax(abs(row))])
    #     return array(res_freqs_model)

    def _model_fast(self, curs, params, plot = False):
        f_r, g = params[:2]
        qubit_params = params[2:]
    #     phis_fine = linspace(phis[0],phis[-1], 1000)
        f_qs = self._qubit_spectrum(curs, *qubit_params)

        freq_span = self._freqs[-1] - self._freqs[0]
        levels = self._eigenlevels(f_qs, f_r, g)

        if plot:
            plt.plot(curs, levels[0,:])
            plt.plot(curs, levels[1,:])
            plt.plot(curs, levels[2,:])
            plt.ylim(self._freqs[0], self._freqs[-1])

        upper_limit = f_r+freq_span
        lower_limit = f_r-freq_span

        res_freqs_model = zeros_like(curs)+0.5*(self._freqs[-1]+self._freqs[0])
        idcs1 = where(logical_and(lower_limit<levels[1,:],
                        levels[1,:]<upper_limit))
        idcs2 = where(logical_and(lower_limit<levels[2,:],
                        levels[2,:]<upper_limit))

        res_freqs_model[idcs1] = levels[1,:][idcs1]
        res_freqs_model[idcs2] = levels[2,:][idcs2]

        return res_freqs_model

    def _brute_cost_function(self, params, curs, res_freqs, period, sws_cur):
        f_res, g, q_max_freq, d = list(params)
        full_params =\
            [f_res, g, period, sws_cur, q_max_freq, d]
        return self._cost_function(full_params, curs,
                                   res_freqs, freqs_fine_number=100)

    def _cost_function(self, params, curs, res_freqs, freqs_fine_number=5e3):

        if self._fast:
            cost = (self._model_fast(curs, params) - res_freqs)**2
        else:
            cost = abs(self._model(curs, params,
                        freqs_fine_number=freqs_fine_number) - res_freqs)
        if self._iteration_counter%100 == 0 and not self._silent:
            clear_output(wait=True)
            print((("{:.4e}, "*len(params))[:-2]).format(*params),
                  "loss:",
                  "%.2f"%(sqrt(sum(cost)/len(curs))/1e6), "MHz")
        self._iteration_counter += 1
        return sum(cost)

    def get_res_points(self):
        return self._res_points













#
# from matplotlib import pyplot as plt
# from IPython.display import clear_output
# from lib2.fulaut.qubit_spectra import *
# from lib2.ResonatorDetector import *
# from lib2.LoggingServer import *
# from lib2.GlobalParameters import *
#
# import scipy
# from scipy import *
# from scipy.optimize import *
# from scipy.signal import *
#
# class AnticrossingOracle():
#
#     """
#     This class automatically processes anticrossing spectral data for
#     different types of qubits and frequency arrangements between the qubits and
#     resonators
#     """
#
#     qubit_spectra = {"transmon":transmon_spectrum}
#
#     def __init__(self, qubit_type, sts_result, plot=False,
#                  fast_res_detect = True, hints = []):
#         self._qubit_spectrum = AnticrossingOracle.qubit_spectra[qubit_type]
#         self._sts_result = sts_result
#         self._plot = plot
#         self._minimum_points_between_zeroes = 5
#         self._minimum_points_around_zero = 5
#         self._distance_from_intersection = 2
#         self._logger = LoggingServer.getInstance()
#         self._fast = True
#         self._fast_res_detect = fast_res_detect
#         self._noisy_data = False
#         self._hints = hints
#         self._iteration_counter = 0
#         self._extract_data()
#
#     def launch(self):
#
#         self._period = self._find_period()
#         potential_sweet_spots = self._find_potential_sweet_spots()
#
#         d_range = slice(0.1, 0.81, 0.9/9)
#         mean_freq = mean(self._res_points[:, 1])
#         f_range = slice(mean_freq-1e6, mean_freq+1.1e6, 1e6)
#         if "fqmax_above" in self._hints:
#             q_freq_range = slice(mean_freq, 12.1e9, 100e6)
#         elif "fqmax_below" in self._hints:
#             q_freq_range = slice(4, mean_freq, 100e6)
#         else:
#             q_freq_range = slice(4e9, 12.1e9, 100e6)
#
#         g_range = slice(20e6, 40.1e6, 20e6/5)
#         Ns = 3
#         args = (self._res_points[:, 0], self._res_points[:, 1])
#         # self._logger.debug(str(res_freq)
#
#         best_fit_loss = 1e100
#         best_fitresult = None
#         # We are not sure where the sweet spot is, so let's choose the best
#         # fit among two possibilities:
#         for sweet_spot_cur in potential_sweet_spots:
#
#             mean_cur = mean(self._res_points[:, 0])
#             distance_to_sws = abs(mean_cur - sweet_spot_cur)
#             shift = round(distance_to_sws / self._period) * self._period * sign(
#                 mean_cur - sweet_spot_cur)
#             sweet_spot_cur += shift
#
#             self._iteration_counter = 0
#             result = brute(self._brute_cost_function,
#                            (f_range, g_range, q_freq_range, d_range), Ns=Ns,
#                            args=args+(self._period, sweet_spot_cur), finish=None)
#
#             freq, g, q_max_freq, d = list(result)
#             full_params = [freq, g, self._period, sweet_spot_cur, q_max_freq, d]
#
#             self._iteration_counter = 0
#             result = minimize(self._cost_function, full_params,
#                               args=args, method="Nelder-Mead")
#
#             loss =\
#                 sqrt(self._cost_function(result.x, *args)/len(self._res_points))/1e6
#
#             brute_loss =\
#                 sqrt(self._cost_function(full_params, *args)/len(self._res_points))/1e6
#
#             if loss < best_fit_loss:
#                 self._brute_opt_params = full_params
#                 self._brute_loss = brute_loss
#                 self._opt_params = result.x
#                 self._loss = loss
#                 best_fit_loss = loss
#                 best_fitresult = result
#
#             if loss<0.05:
#                 break
#
#         res_freq, g, period, sweet_spot_cur, q_freq, d = best_fitresult.x
#
#         if self._plot:
#             plt.figure()
#             plt.plot(self._res_points[:, 0], self._res_points[:, 1], '.',
#                         label="Data")
#             plt.plot([sweet_spot_cur], [mean(self._res_points[:, 1])], '+')
#
#             # p0 = [mean((res_freq_2, res_freq_2)), 0.03, period,
#             #                                         sweet_spot_cur, 10, 0.6]
#             # plt.plot(self._curs, self._model(self._curs, p0), "o")
#             plt.plot(self._res_points[:, 0],
#                      self._model_fast(self._res_points[:, 0],
#                      self._brute_opt_params, False),
#                      "yellow", ls=":", label="Brute")
#             plt.plot(self._res_points[:, 0],
#                      self._model_fast(self._res_points[:, 0],
#                      best_fitresult.x, False),
#                      "orange", ls="-", marker=".", label="Final")
#             plt.legend()
#             plt.gcf().set_size_inches(15, 5)
#
#         best_fitresult.x[0] = best_fitresult.x[0]
#         best_fitresult.x[4] = best_fitresult.x[4]
#         return best_fitresult.x, best_fit_loss
#
#     def _extract_data(self, plot=False):
#         try:
#             data = self._sts_result.get_data()
#         except:
#             # maybe we have raw dict
#             data = self._sts_result
#         try:
#             curs = data["Current [A]"]
#         except:
#             curs = data["current"]
#
#         try:
#             freqs = data["frequency"]
#         except:
#             freqs = data["Frequency [Hz]"]
#
#         self._data = data["data"]
#
#         data = self._data
#         res_freqs = []
#
#         def comlex_ptp_estimation(Z):
#             point0 = Z[0, 0]
#             point1 = Z.ravel()[argmax(abs(Z-point0))]
#             point2 = Z.ravel()[argmax(abs(Z-point1))]
#             point3 = Z.ravel()[argmax(abs(Z-point2))]
#             return abs(point3-point2)
#             # point0 = Z[0, 0]
#             # point1 = Z.ravel()[argmax(angle(Z-point0))]
#             # point2 = Z.ravel()[argmax(angle(Z-point1))]
#             # point3 = Z.ravel()[argmax(angle(Z-point2))]
#             # return angle(point3-point2)
#
#         mean_derivative = mean(abs(diff(data)))
#         # mean_derivative = mean(angle(diff(data)))
#         data_ptp = comlex_ptp_estimation(data)
#
#         if mean_derivative > 0.2*data_ptp:
#             # we probably have a lot of noise
#             self._noisy_data = True
#
#         preprocessed_data = filtered_data if self._noisy_data else data
#         self._preprocessed_data = preprocessed_data
#
#         # Taking peaks deeper than half of the distance between the median
#         # transmission level and the deepest point
#         threshold = abs(preprocessed_data).min()+\
#                 0.5*(median(abs(preprocessed_data))-abs(preprocessed_data).min())
#         # threshold = angle(preprocessed_data).min()+\
#         #         0.5*(median(angle(preprocessed_data))-angle(preprocessed_data).min())
#
#         res_points = []
#         self._extracted_indices = []
#         self._extraction_types =[]
#
#         for idx, preprocessed_row in enumerate(preprocessed_data):
#             preprocessed_row = abs(preprocessed_row)
#             # preprocessed_row = angle(preprocessed_row)
#             extrema = argrelextrema(preprocessed_row, less, order=10)[0]
#             extrema = extrema[preprocessed_row[extrema]<threshold]
#
#             if len(extrema) > 0:
#                 if GlobalParameters().resonator_types['reflection'] == True:
#                     RD = ResonatorDetector(freqs, preprocessed_data[idx], plot=False,
#                                        fast=self._fast_res_detect, type='reflection')
#                 else:
#                     RD = ResonatorDetector(freqs, preprocessed_data[idx], plot=False,
#                                        fast=self._fast_res_detect, type='transmission')
#                 result = RD.detect()
#                 if result is not None:
#                     res_points.append((curs[idx], result[0]))
#                     self._extraction_types.append("fit")
#                     self._extracted_indices.append(idx)
#
#                 else:
#                     smallest_extremum =\
#                         extrema[argmin(preprocessed_row[extrema])]
#                     res_points.append((curs[idx], freqs[smallest_extremum]))
#                     self._extraction_types.append("min")
#                     self._extracted_indices.append(idx)
#
#
#         self._res_points = array(res_points)
#         self._freqs = freqs
#         self._curs = curs
#
#         # if self._plot:
#         #     plt.figure()
#         #     plt.plot(self._res_points[:,0], self._res_points[:,1], 'C1.',
#         #                 label="Extracted points")
#         #     plt.pcolormesh(curs, freqs, abs(data).T)
#         #     plt.legend()
#         #     plt.gcf().set_size_inches(15,5)
#
#     def _find_period(self):
#         extracted_no_mean = self._res_points[:,1]-mean(self._res_points[:,1])
#         extracted_zero_padded = scipy.zeros(len(self._curs))
#         extracted_zero_padded[self._extracted_indices] = extracted_no_mean
#         data = extracted_zero_padded
#
#         self._corr = corr = correlate(data, data, "full")[data.size-1:]
#         peaks = argrelextrema(corr, greater, order=10)[0]
#         try:
#             period = peaks[argmax(corr[peaks])]
#             print(peaks, period)
#             return self._curs[period] - self._curs[0]
#         except ValueError:
#             return 1.5 * ptp(self._curs)
#
#
#     def _model_square(self, duty, phase, x):
#         return square(2*pi*x/self._period-phase, duty)
#
#     def _cost_function_sweet_spots(self, p, x, y):
#         duty, phase = p
#         fit_data = self._model_square(duty, phase, x)
#         return -sum(fit_data*y)
#
#     def _find_potential_sweet_spots(self):
#         data = self._res_points[:,1]-mean(self._res_points[:,1])
#         duty, phase = brute(self._cost_function_sweet_spots,
#                             ((0, 1), (-pi, pi)),
#                             Ns = 50,
#                             args=(self._res_points[:,0], data),
#                             full_output=0)
#         self._duty, self._phase = duty, phase
#         sws1 = phase/2/pi*self._period + self._period*duty/2
#         sws2 = phase/2/pi*self._period - self._period*(1-duty)/2
#
#         if max(abs(diff(data))) > 0.5*ptp(data):
#             # we probably have anticrossings
#             max_num_of_anticrossings = ceil(ptp(self._curs)/self._period)*2
#             min_number_of_anticrossings = floor(ptp(self._curs)/self._period)*2
#             large_derivatives = where(abs(diff(data)) > 0.5*ptp(data))[0]
#
#             if min_number_of_anticrossings <= len(large_derivatives) <= max_num_of_anticrossings:
#                 # everything is fine, not noise
#                 return [sws2]
#
#         elif max(abs(diff(data))) < 0.1*ptp(data):
#             # we probably have smooth curves
#             return [sws1]
#
#         # we will check both, noisy scan
#         return sws1, sws2
#
#
#     # def find_resonator_intersections(self):
#     #     plt.plot(self._res_points[:,0],
#     #              ones_like(self._res_points[:,1])*mean(self._res_points[:,1]),
#     #              'C2', label="Mean")
#     #     data = (mean(self._res_points[:,1])-self._res_points[:,1])
#     #
#     #     raw_intersections = where((data[:-1]*data[1:])<0)[0]+1
#     #
#     #     refined_intersections = []
#     #     current_intersection = raw_intersections[0]
#     #     refined_intersections.append(current_intersection)
#     #
#     #     for raw_intersection in raw_intersections[1:]:
#     #         distance = raw_intersection - current_intersection
#     #         if distance > self._minimum_points_between_zeroes:
#     #             current_intersection = raw_intersection
#     #             refined_intersections.append(current_intersection)
#     #
#     #     fine_intersections = []
#     #
#     #     points_around_zero = self._minimum_points_around_zero
#     #     distance_from_intersection = self._distance_from_intersection
#     #     for intersection in refined_intersections:
#     #         if intersection <= distance_from_intersection\
#     #                     or  intersection>=(len(data)-distance_from_intersection):
#     #             continue
#     #
#     #         if points_around_zero-1 >= intersection:
#     #             left_edge = 0
#     #             right_edge = intersection+points_around_zero
#     #
#     #         elif points_around_zero-1 < \
#     #                 intersection<\
#     #                     (len(data)-points_around_zero):
#     #             left_edge = intersection-points_around_zero
#     #             right_edge = intersection+points_around_zero
#     #         else:
#     #             right_edge = len(data)
#     #             left_edge = intersection-points_around_zero
#     #
#     #         left_points = data[left_edge:intersection-distance_from_intersection]
#     #         right_points = data[intersection+distance_from_intersection:right_edge]
#     #
#     #         self._logger.debug("Intersection: "+str(intersection)\
#     #                             +", points: "+str((left_points, right_points)))
#     #
#     #         if len(set(left_points<0))==1\
#     #             and len(set(right_points<0))==1\
#     #             and left_points[0]*right_points[0]<0:
#     #             fine_intersections.append(intersection)
#     #
#     #     self._logger.debug("Fine intersections: "+str(fine_intersections))
#     #     return fine_intersections
#
#     # @staticmethod
#     # def _transmission(f_q, f_probe, f_r, g):
#     #     κ = 1e6
#     #     γ = 1e6
#     #     return abs(1/(κ+1j*(f_r-f_probe)+(g**2)/(γ+1j*(f_q-f_probe))))**2
#
#     @staticmethod
#     def _eigenlevels(f_q, f_r, g):
#         E0 = (f_r - f_q)/2
#         E1 = f_r-1/2*sqrt(4*g**2+(f_q-f_r)**2)
#         E2 = f_r+1/2*sqrt(4*g**2+(f_q-f_r)**2)
#         return array([E0-E0, E1-E0, E2-E0])
#
#
#     # def _model(self, curs, params, plot_colours = False, freqs_fine_number=5e3):
#     #
#     #     f_r, g = params[:2]
#     #     qubit_params = params[2:]
#     # #     phis_fine = linspace(phis[0],phis[-1], 1000)
#     #     f_qs = self._qubit_spectrum(curs, *qubit_params)
#     #
#     #     span = ptp(self._freqs)
#     #     freqs_fine = linspace(self._freqs[0]-span*0.5,
#     #                             self._freqs[-1]+span*0.5,
#     #                                 freqs_fine_number)
#     #
#     #     XX, YY = meshgrid(f_qs, freqs_fine)
#     #     transmissions = self._transmission(XX, YY, f_r, g)
#     #
#     #     if plot_colours:
#     #         plt.pcolormesh(curs, freqs_fine, transmissions)
#     #         plt.colorbar()
#     #
#     #     res_freqs_model = []
#     #     for row in transmissions.T:
#     #         res_freqs_model.append(freqs_fine[argmax(abs(row))])
#     #     return array(res_freqs_model)
#
#     def _model_fast(self, curs, params, plot = False):
#         f_r, g = params[:2]
#         qubit_params = params[2:]
#     #     phis_fine = linspace(phis[0],phis[-1], 1000)
#         f_qs = self._qubit_spectrum(curs, *qubit_params)
#
#         freq_span = self._freqs[-1] - self._freqs[0]
#         levels = self._eigenlevels(f_qs, f_r, g)
#
#         if plot:
#             plt.plot(curs, levels[0,:])
#             plt.plot(curs, levels[1,:])
#             plt.plot(curs, levels[2,:])
#             plt.ylim(self._freqs[0], self._freqs[-1])
#
#         upper_limit = f_r+freq_span
#         lower_limit = f_r-freq_span
#
#         res_freqs_model = zeros_like(curs)+0.5*(self._freqs[-1]+self._freqs[0])
#         idcs1 = where(logical_and(lower_limit<levels[1,:],
#                         levels[1,:]<upper_limit))
#         idcs2 = where(logical_and(lower_limit<levels[2,:],
#                         levels[2,:]<upper_limit))
#
#         res_freqs_model[idcs1] = levels[1,:][idcs1]
#         res_freqs_model[idcs2] = levels[2,:][idcs2]
#
#         return res_freqs_model
#
#     def _brute_cost_function(self, params, curs, res_freqs, period, sws_cur):
#         f_res, g, q_max_freq, d = list(params)
#         full_params =\
#             [f_res, g, period, sws_cur, q_max_freq, d]
#         return self._cost_function(full_params, curs,
#                                    res_freqs, freqs_fine_number=100)
#
#     def _cost_function(self, params, curs, res_freqs, freqs_fine_number=5e3):
#
#         if self._fast:
#             cost = (self._model_fast(curs, params) - res_freqs)**2
#         else:
#             cost = abs(self._model(curs, params,
#                         freqs_fine_number=freqs_fine_number) - res_freqs)
#         if self._iteration_counter%100 == 0:
#             clear_output(wait=True)
#             print((("{:.4e}, "*len(params))[:-2]).format(*params),
#                   "loss:",
#                   "%.2f"%(sqrt(sum(cost)/len(curs))/1e6), "MHz")
#         self._iteration_counter += 1
#         return sum(cost)
#
#     def get_res_points(self):
#         return self._res_points
