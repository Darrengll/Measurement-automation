from matplotlib import pyplot as plt
from scipy import *
import numpy as np
from scipy.signal import *
from scipy.optimize import *
from IPython.display import clear_output
from operator import itemgetter

from lib2.fulaut.qubit_spectra import *

import fnmatch
import os
import pickle
from threading import Lock
from datetime import datetime

from IPython.display import clear_output
import locale


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


class SpectrumOracle:
    """
    This class automatically processes spectral data for different types of qubits
    """
    qubit_spectra = {"transmon": transmon_spectrum}

    def __init__(self, qubit_type, tts_result, initial_guess_qubit_params,
                 plot=False):
        """
        parameter_period_grid = grids[0]
        parameter_at_sweet_spot_grid = grids[1]
        frequency_grid = grids[2]
        d_grid = grids[3]
        alpha_grid = grids[4]
        """
        self._tts_result = tts_result
        self._qubit_spectrum = SpectrumOracle.qubit_spectra[qubit_type]
        self._plot = plot
        self._extract_data()

        self._y_scan_area_size = 10e-3
        period, sweet_spot_cur, q_freq, d = initial_guess_qubit_params
        q_freq = q_freq / 1e9

        period_grid = period, period, 1
        sws_grid = sweet_spot_cur - 0.05 * period, sweet_spot_cur + 0.05 * period, 30
        freq_grid = 2, 10, 501
        d_grid = .1, .9, 50
        half_alpha_grid = 100e-3, 150e-3, 10

        slices = []
        self._grids = (period_grid, sws_grid, freq_grid, d_grid, half_alpha_grid)
        for grid in self._grids:
            if grid[2] != 1:
                step = (grid[1] - grid[0]) / grid[2]
                slices.append(slice(grid[0], grid[1] + 0.01 * step, step))
            else:
                slices.append(slice(grid[0], grid[0] + 0.01, 1))

        self._p0 = [(grid[0] + grid[1]) / 2 for grid in self._grids]
        self._slices = slices

        if self._plot:
            plt.figure("Spectrum oracle fit of %s" % self._tts_result.get_name())

            plt.plot(self._points[:, 0], self._points[:, 1], ".", markersize=2, label="Extracted points")
            plt.plot([self._grids[1][0]], [mean(self._points[:, 1])],
                     "|", markersize=100)
            plt.plot([self._grids[1][1]], [mean(self._points[:, 1])],
                     "|", markersize=100)

            period = self._p0[0]
            plt.plot([self._grids[1][0] + period / 2], [mean(self._points[:, 1])],
                     "|", markersize=100)
            plt.plot([self._grids[1][1] + period / 2], [mean(self._points[:, 1])],
                     "|", markersize=100)

        self._coarse_brute_candidates = []
        self._coarse_brute_nop_ranking = []
        self._coarse_brute_distance_ranking = []

        self._fine_brute_candidates = []
        self._fine_brute_nop_ranking = []
        self._fine_brute_distance_ranking = []

    def set_start_datetime(self, datetime):
        self._datetime = datetime

    def get_start_datetime(self):
        return self._datetime

    def launch(self):

        freq_slice = self._slices[2]
        d_slice = self._slices[3]
        sws_slice = self._slices[1]
        period_slice = self._slices[0]
        self._iterations = (self._grids[2][2] + 1) * (
                self._grids[3][2] + 1)  # * (self._grids[1][2] + 1) *

        args = (self._y_scan_area_size, self._points)

        # refine frequency
        self._counter = 0
        self._coarse_brute_candidates = []
        self._coarse_brute_nop_ranking = []
        self._coarse_brute_distance_ranking = []
        self._coarse_brute_frequency_ranking = []

        brute(self._cost_function_coarse,
              (period_slice, sws_slice, freq_slice, d_slice),
              args=(self._y_scan_area_size,
                    self._points),
              full_output=False,
              finish=None)

        very_coarse_ranking = \
            sorted(zip(self._coarse_brute_nop_ranking,
                       self._coarse_brute_distance_ranking,
                       self._coarse_brute_candidates),
                   key=itemgetter(0, 1))
        self._very_coarse_candidates = [rank[2] for rank in very_coarse_ranking]
        self._very_coarse_nops = [rank[0] for rank in very_coarse_ranking]

        self._opt_params_very_coarse = self._very_coarse_candidates[0]

        if self._plot:
            ptp_nop = ptp(1 / array(self._very_coarse_nops[:20]))
            min_nop = 1 / max(self._very_coarse_nops[:20])
            for idx, opt_params, nop in zip(range(20), self._very_coarse_candidates[:20], self._very_coarse_nops[:20]):
                plt.plot(self._parameter_values,
                         self._qubit_spectrum(self._parameter_values,
                                              *opt_params), ":",
                         color=(1 - (1 / nop - min_nop) / ptp_nop, 0, (1 / nop - min_nop) / ptp_nop), lw=.5,
                         label="Very coarse search" if idx == 0 else None)

        freq_slice = slice(self._opt_params_very_coarse[2] - 200e-3,
                           self._opt_params_very_coarse[2] + 201e-3,
                           10e-3)
        d_slice = slice(self._opt_params_very_coarse[3] - 0.1,
                        self._opt_params_very_coarse[3] + 0.101,
                        0.02)

        self._refine_freq_slice = freq_slice
        self._iterations = (self._grids[1][2] + 1) * 21 * 11

        self._counter = 0
        self._coarse_brute_candidates = []
        self._coarse_brute_nop_ranking = []
        self._coarse_brute_distance_ranking = []
        self._coarse_brute_frequency_ranking = []

        brute(self._cost_function_coarse,
              (period_slice, sws_slice, freq_slice, d_slice),
              args=args,
              full_output=True,
              finish=None)

        nop_rank, mean_dist, opt_params_coarse = \
            sorted(zip(self._coarse_brute_nop_ranking,
                       self._coarse_brute_distance_ranking,
                       self._coarse_brute_candidates),
                   key=itemgetter(0, 1))[0]

        self._coarse_brute_loss = (nop_rank, mean_dist)
        chosen_points = \
            self._cost_function_coarse(opt_params_coarse, self._y_scan_area_size,
                                       self._points, True)[1]
        result = minimize(self._cost_function_coarse, opt_params_coarse,
                          args=(self._y_scan_area_size, chosen_points), method="Nelder-Mead")
        opt_params_coarse = result.x
        self._opt_params_coarse = opt_params_coarse

        # extrema = self._argrelextrema2D(loss).T
        # self._candidate_freqs = mgrid[freq_slice][extrema[0]]
        # self._candidate_ds = mgrid[d_slice][extrema[1]]
        # self._candidate_losses = loss[extrema[0], extrema[1]]
        # return None
        # opt_params_coarse = max(self._candidate_freqs), min(self._candidate_ds)
        self._opt_params_coarse = opt_params_coarse
        self._coarse_frequency = opt_params_coarse[2]

        if self._plot:
            plt.plot(self._parameter_values,
                     self._qubit_spectrum(self._parameter_values,
                                          *opt_params_coarse), ":", label="Coarse result")

        fine_period_grid = slice(opt_params_coarse[0], opt_params_coarse[0] + 0.1, 1)
        fine_sws_grid = slice(opt_params_coarse[1], opt_params_coarse[1] + 0.1,
                              1)
        fine_freq_grid = slice(self._coarse_frequency,
                               self._coarse_frequency + 0.41, 0.4 / 20)
        fine_d_grid = slice(opt_params_coarse[3],
                            opt_params_coarse[3] + .1, 1)
        fine_half_alpha_grid = self._slices[-1]
        self._fine_slices = (fine_period_grid,
                             fine_sws_grid,
                             fine_freq_grid,
                             fine_d_grid,
                             fine_half_alpha_grid)
        self._iterations = 21 * (self._grids[-1][2] + 1)

        self._counter = 0
        self._fine_brute_candidates = []
        self._fine_brute_nop_ranking = []
        self._fine_brute_distance_ranking = []

        brute(self._cost_function_fine_fast, self._fine_slices,
              args=args, Ns=1,
              full_output=False, finish=None)

        self._fine_ranking = sorted(zip(self._fine_brute_nop_ranking,
                                        self._fine_brute_distance_ranking,
                                        self._fine_brute_candidates),
                                    key=itemgetter(0, 1))

        self._fine_brute_candidates = [rank[2] for rank in self._fine_ranking]
        self._fine_brute_nop_ranking = [rank[0] for rank in self._fine_ranking]
        best_solution = self._fine_ranking[0][-1]

        if self._plot:
            # ptp_nop = ptp(1 / array(self._fine_brute_nop_ranking[:]))
            # min_nop = 1 / max(self._fine_brute_nop_ranking[:])
            # for idx, opt_params, nop in zip(range(len(self._fine_ranking)),
            #                                 self._fine_brute_candidates[:],
            #                                 self._fine_brute_nop_ranking[:]):
            #     plt.plot(self._parameter_values[::5],
            #              self._qubit_spectrum(self._parameter_values[::5],
            #                                   *opt_params[:-1]), ":", marker = ".",
            #              color=(1 - (1 / nop - min_nop) / ptp_nop, 0, (1 / nop - min_nop) / ptp_nop), lw=.5,
            #              label="Fine search" if idx == 0 else None)

            plt.plot(self._parameter_values,
                     self._qubit_spectrum(self._parameter_values,
                                          *best_solution[:-1]), "-",
                     color=(1 - (1 / nop - min_nop) / ptp_nop, 0, (1 / nop - min_nop) / ptp_nop),
                     label="Fine result")

        chosen_points = self._cost_function_fine_fast(best_solution, self._y_scan_area_size,
                                                      self._points, True)[1]
        if self._plot:
            for idx, branch in enumerate(chosen_points):
                if branch.size > 0:
                    plt.plot(branch[:, 0], branch[:, 1], "C%d." % idx)

        all_chosen = concatenate([points for points in chosen_points if points.size > 0])
        result = minimize(self._cost_function_fine_fast, best_solution,
                          args=(self._y_scan_area_size, all_chosen), method="Nelder-Mead")
        best_solution = result.x

        self._fine_opt_params = best_solution
        self._fine_frequency = best_solution[2]
        self._fine_half_alpha = best_solution[4]

        self._counter = 0
        self._final_loss = self._cost_function_fine_fast(self._fine_opt_params,
                                                         *args)

        if self._plot:

            for i in range(0, 4):
                plt.plot(self._parameter_values,
                         self._qubit_spectrum(self._parameter_values,
                                              *self._fine_opt_params[:-1]) -
                         i * self._fine_opt_params[-1], color="C%d" % i)
            plt.legend()
            plt.xlim(self._parameter_values[0], self._parameter_values[-1])
            plt.ylim(self._frequencies[0], self._frequencies[-1])
            plt.gcf().set_size_inches(15, 5)

        self._fine_opt_params[2] = self._fine_opt_params[2] * 1e9
        self._fine_opt_params[-1] = self._fine_opt_params[-1] * 1e9 * 2  # half alpha to alpha

        return array(self._fine_opt_params)

    def _extract_data(self):
        try:
            parameter_name = self._tts_result._parameter_names[0]
        except:
            parameter_name = "Current [A]"
        try:
            data = self._tts_result.get_data()
        except AttributeError:
            data = self._tts_result
        self._parameter_values = data[parameter_name]
        try:
            self._frequencies = data["Frequency [Hz]"][:] / 1e9
        except:
            self._frequencies = data["frequency"][:] / 1e9

        self._freq_resolution = self._frequencies[1] - self._frequencies[0]
        self._Z = (data["data"].T - mean(data["data"], -1)).T

        self._threshold = sqrt(median(diff(abs(self._Z)) ** 2))

        points = []
        for idx in range(len(self._parameter_values)):
            row = abs(self._Z)[idx]
            row = row - median(row)

            if sqrt(median(diff(row) ** 2)) > 0.1 * ptp(row):
                # we probably have a noisy row
                continue

            bright_extrema = find_peaks(row, prominence=0.25 * ptp(row))[0]
            bright_extrema = bright_extrema[
                row[bright_extrema] > 1 * self._threshold]

            sort_idx = argsort(row[bright_extrema])
            bright_extrema = bright_extrema[sort_idx][-4:]  # only take up to 4 brightest ones
            points += list(
                zip([self._parameter_values[idx]] * len(bright_extrema),
                    self._frequencies[bright_extrema]))

        self._points = array(points)

        for point in points:
            very_close_y_indices = \
                where(abs(self._points[:, 1] - point[1]) < self._freq_resolution)[0]
            if len(very_close_y_indices) > 20:
                self._points = delete(self._points, very_close_y_indices,
                                      0).reshape(-1, 2)

        if self._plot:
            plt.figure("Spectrum oracle extraction")
            x = self._parameter_values
            freqs = self._frequencies
            x_plot = concatenate(
                (x - (x[1] - x[0]) / 2, x[-1:] + (x[1] - x[0]) / 2))
            freqs_plot = concatenate((freqs - (freqs[1] - freqs[0]) / 2,
                                      freqs[-1:] + (freqs[1] - freqs[0]) / 2))

            plt.pcolormesh(x_plot, freqs_plot, abs(self._Z).T)
            plt.plot(self._points[:, 0], self._points[:, 1], 'r.', markersize=2)
            plt.colorbar()
            plt.gcf().set_size_inches(15, 5)

    def _cost_function_coarse(self, params, y_scan_area_size, points,
                              verbose=False):
        percentage_done = self._counter / self._iterations * 100

        self._counter += 1
        #         print(params)

        distances = abs(self._qubit_spectrum(points[:, 0], *params) - points[:, 1])
        chosen = distances < y_scan_area_size
        distances_chosen = distances[chosen]
        chosen_points = points[chosen]

        d = params[3]
        if len(chosen_points) < len(set(self._points[:, 0])) * .75 or d > 0.9:
            mean_distance = sum(distances) ** 2 / len(distances)
            total_nop = 0.1
            # if self._counter % 1 == 0:
            #     print("\rDone: %.2f%%, %.d/%d" % (
            #         percentage_done, self._counter, self._iterations), end="")
            #     print(", [" + (("{:.2e}, " * len(params))[:-2]).format(
            #         *params) + "]", end="")
            #     print(", mean_dist:", "%.2e" % mean_distance,
            #           ", chosen points:", total_nop)
        else:
            mean_distance = distances_chosen.sum() ** 2 / len(chosen_points)
            bin = round(len(set(self._points[:, 0])) * 0.1, 0)
            total_nop = round(len(distances_chosen) / bin, 0) * bin

            self._coarse_brute_candidates.append(params)
            self._coarse_brute_nop_ranking.append(1 / total_nop)
            self._coarse_brute_distance_ranking.append(mean_distance)

            if percentage_done <= 100 and self._counter % 10 == 0:
                print("\rDone: %.2f%%, %.d/%d" % (
                    percentage_done, self._counter, self._iterations), end="")
                print(", [" + (("{:.2e}, " * len(params))[:-2]).format(
                    *params) + "]", end="")
                print(", mean_dist:", "%.2e" % mean_distance,
                      ", chosen points:", total_nop)
                clear_output(wait=True)

            elif self._counter % 10 == 0:
                print("\rDone: 100%, polishing...", end="")
                print(", [" + (("{:.2e}, " * len(params))[:-2]).format(
                    *params) + "]", end="")
                print(", mean_dist:", "%.2e" % mean_distance,
                      ", chosen points:", total_nop)
                clear_output(wait=True)

        loss_value = sum(distances) ** 2 / len(distances)

        if verbose:
            return loss_value, chosen_points

        return loss_value

    def _cost_function_fine_fast(self, params, y_scan_area_size, points,
                                 verbose=False):

        percentage_done = self._counter / self._iterations * 100
        if percentage_done <= 100 and self._counter % 10 == 0:
            print("\rDone: %.2f%%, %.d/%d" % (
                percentage_done, self._counter, self._iterations), end="")
            print(
                ", [" + (("{:.3e}, " * len(params))[:-2]).format(*params) + "]",
                end="")
        elif self._counter % 10 == 0:
            print("\rDone: 100%, polishing...", end="")
            print(
                ", [" + (("{:.3e}, " * len(params))[:-2]).format(*params) + "]",
                end="")
        self._counter += 1

        q_freqs = self._qubit_spectrum(points[:, 0], *params[:4])

        frequency_shifts = [0, -params[4], -2 * params[4], -3 * params[4]]
        loss_factors = [1, 1, 1, 1]
        lines_chosen_distances = []
        lines_chosen_points = []
        lines_distances = []

        for shift in frequency_shifts:
            distances = abs(q_freqs - points[:, 1] + shift)
            lines_distances.append(distances)
            chosen = distances < y_scan_area_size
            close_distances = distances[chosen]
            close_points = points[chosen]

            nonzerodiff = flatnonzero(
                np.diff(np.r_[0, points[chosen][:, 0], 0]))
            groups_of_same_x_coord = vstack((nonzerodiff[:-1], nonzerodiff[1:])) \
                .T

            chosen_distances = []
            chosen_points = []
            for group in groups_of_same_x_coord:
                same_x_distances = close_distances[group[0]:group[1]]
                argmin_of_group = argmin(same_x_distances)
                chosen_distances.append(same_x_distances[argmin_of_group])
                chosen_points.append(close_points[group[0] + argmin_of_group])

            lines_chosen_distances.append(chosen_distances)
            lines_chosen_points.append(chosen_points)

        if len(lines_chosen_distances[0]) < 0.75 * len(set(self._points[:, 0])):
            mean_distance = sum(lines_distances[0]) ** 2 / len(
                lines_distances[0])
            total_nop = 0.1
            self._fine_brute_distance_ranking.append(mean_distance)
            self._fine_brute_nop_ranking.append(1 / total_nop)
            self._fine_brute_candidates.append(params)
        else:
            all_distances = concatenate(lines_chosen_distances)
            bin = round(len(set(self._points[:, 0])) * 0.25, 0)
            total_nop = round(len(all_distances) / bin, 0) * bin
            mean_distance = sum(all_distances ** 2) / len(
                lines_chosen_distances[0])

            self._fine_brute_distance_ranking.append(mean_distance)
            self._fine_brute_nop_ranking.append(1 / total_nop)
            self._fine_brute_candidates.append(params)
            if self._counter % 10 == 0:
                # print(", loss:", "%.2e" % loss_value,
                #       ", chosen points:", len(concatenate(lines_chosen_distances)))
                print(", dist:%.2e" % sqrt(mean_distance),
                      ", chosen points: %d" % total_nop,
                      ", bin: %d" % bin)
                clear_output(wait=True)

        loss_value = sqrt(mean_distance) + 1 / total_nop

        if verbose:
            # print(len(concatenate(lines_chosen_distances)))
            return loss_value, [array(l) for l in lines_chosen_points]

        return loss_value

    def _cost_function_fine(self, params, y_scan_area_size, points,
                            verbose=False):

        x_coords = array(list(sorted(set(points[:, 0]))))

        loss = []
        loss2 = []
        loss3 = []
        chosen_points = []
        chosen_points2 = []
        chosen_points3 = []

        total_distance_of_the_main_line_from_points = 0
        q_freqs = self._qubit_spectrum(x_coords, *params[:4])

        for idx, x_coord in enumerate(x_coords):
            same_x_points = points[points[:, 0] == x_coord]
            distances = abs(q_freqs[idx] - same_x_points[:, 1])
            distances2 = abs(q_freqs[idx] - params[-1] - same_x_points[:, 1])
            distances3 = abs(
                q_freqs[idx] - 2 * params[-1] - same_x_points[:, 1])
            total_distance_of_the_main_line_from_points += sum(distances)

            min_arg = argmin(distances)
            min_arg_2 = argmin(distances2)
            min_arg_3 = argmin(distances3)
            min_dist = distances[min_arg]
            min_dist_2 = distances2[min_arg_2]
            min_dist_3 = distances3[min_arg_3]

            if min_dist < y_scan_area_size:
                loss.append(min_dist)
                if verbose:
                    chosen_points.append((x_coord, same_x_points[min_arg, 1]))
            if min_dist_2 < y_scan_area_size:
                loss2.append(min_dist_2)
                if verbose:
                    chosen_points2.append(
                        (x_coord, same_x_points[min_arg_2, 1]))
            if min_dist_3 < y_scan_area_size:
                loss3.append(min_dist_3)
                if verbose:
                    chosen_points3.append(
                        (x_coord, same_x_points[min_arg_3, 1]))

        total_chosen_points = (len(loss) + len(loss2) + len(loss3))

        if len(loss) < len(x_coords) / 3:
            loss_value = total_distance_of_the_main_line_from_points ** 2
        else:
            loss_value = sum(array(loss)) / (len(loss) + 1) + \
                         0.1 * sum(array(loss2)) / (len(loss2) + 1) + \
                         0.01 * sum(array(loss3)) / (len(loss3) + 1)
            loss_value /= total_chosen_points ** 2

        if self._counter % 10 == 0:
            percentage_done = self._counter / self._iterations * 100
            clear_output(wait=True)
            print("\rDone: %.2f%%, %.d/%d" % (
                percentage_done, self._counter, self._iterations), end="")
            print(
                ", [" + (("{:.2e}, " * len(params))[:-2]).format(*params) + "]",
                end="")
            print(", loss:", "%.2e" % loss_value, ", chosen points:",
                  (total_chosen_points))
        self._counter += 1

        if verbose:
            return loss_value, (
                array(chosen_points), array(chosen_points2), array(chosen_points3))
        return loss_value

    def _argrelextrema2D(self, data):
        both_axes_extrema = []
        y = array(argrelextrema(data, less, 1, order=10)).T
        x = array(argrelextrema(data, less, 0, order=10)).T
        for point in y:
            equality = (point == x).T
            equals = where(np.logical_and(*equality))[0]
            if equals.size != 0:
                both_axes_extrema.append(point)
        return array(both_axes_extrema)

    def visualize(self):
        self._fine_opt_params[2] = self._fine_opt_params[2] / 1e9
        self._fine_opt_params[-1] = self._fine_opt_params[-1] / 1e9

        plt.figure("Spectrum oracle: %s" % self._tts_result.get_name())
        plt.subplot(2, 1, 1)

        x = self._parameter_values
        freqs = self._frequencies
        x_plot = concatenate(
            (x - (x[1] - x[0]) / 2, x[-1:] + (x[1] - x[0]) / 2))
        freqs_plot = concatenate((freqs - (freqs[1] - freqs[0]) / 2,
                                  freqs[-1:] + (freqs[1] - freqs[0]) / 2))

        plt.pcolormesh(x_plot, freqs_plot, abs(self._Z).T, rasterized=True)
        plt.plot(self._points[:, 0], self._points[:, 1], 'r.')
        z = plt.colorbar()
        plt.xlabel('Current [A]')
        plt.ylabel('Frequency [GHz]')
        z.set_label('S21', rotation=90)
        plt.gcf().set_size_inches(10, 10)

        plt.subplot(2, 1, 2)

        plt.plot(self._points[:, 0], self._points[:, 1], ".")
        plt.plot([self._grids[1][0]], [mean(self._points[:, 1])],
                 "|", markersize=100)
        plt.plot([self._grids[1][1]], [mean(self._points[:, 1])],
                 "|", markersize=100)
        plt.plot(self._parameter_values,
                 self._qubit_spectrum(self._parameter_values,
                                      *self._opt_params_coarse), ":")
        plt.plot(self._parameter_values,
                 self._qubit_spectrum(self._parameter_values,
                                      *self._fine_opt_params[:-1]))
        plt.plot(self._parameter_values,
                 self._qubit_spectrum(self._parameter_values,
                                      *self._fine_opt_params[:-1]) -
                 self._fine_opt_params[-1] / 2)
        plt.plot(self._parameter_values,
                 self._qubit_spectrum(self._parameter_values,
                                      *self._fine_opt_params[:-1]) - 2 *
                 self._fine_opt_params[-1] / 2)
        plt.plot(self._parameter_values,
                 self._qubit_spectrum(self._parameter_values,
                                      *self._fine_opt_params[:-1]) - 3 *
                 self._fine_opt_params[-1] / 2)
        plt.xlabel('Current [A]')
        plt.ylabel('Frequency [GHz]')
        plt.gcf().set_size_inches(10, 10)

        plt.tight_layout()
        plt.show()

        data = {'extracted points': self._points, 'parameters': self._parameter_values,
                'fine opt parameters': self._fine_opt_params, 'opt params coarse': self._opt_params_coarse}
        self._data = data

    def save(self):

        self.visualize()
        name = self._tts_result._name + "-oracle-fit"
        with open(os.path.join(self.get_save_path(), name + '.pkl'), 'w+b') as f:
            pickle.dump(self, f)
        with open(os.path.join(self.get_save_path(), name + '_raw_data.pkl'), 'w+b') as f:
            pickle.dump(self._data, f)
        with open(os.path.join(self.get_save_path(), name + '_context.txt'), 'w+') as f:
            f.write(str(self._fine_opt_params))

        plt.savefig(os.path.join(self.get_save_path(), name + ".png"), bbox_inches='tight')
        plt.savefig(os.path.join(self.get_save_path(), name + ".pdf"), bbox_inches='tight')
        plt.close("Spectrum oracle: %s" % self._tts_result.get_name())

    def get_save_path(self):

        if not os.path.exists("data"):
            os.makedirs("data")

        sample_directory = os.path.join('data', self._tts_result._sample_name)
        if not os.path.exists(sample_directory):
            os.makedirs(sample_directory)

        locale.setlocale(locale.LC_TIME, "C")
        date_directory = os.path.join(sample_directory,
                                      self._tts_result.get_start_datetime().strftime("%b %d %Y"))
        if not os.path.exists(date_directory):
            os.makedirs(date_directory)

        time_directory = os.path.join(date_directory,
                                      self._tts_result.get_start_datetime().strftime("%H-%M-%S")
                                      + " - " + self._tts_result._name + "-oracle-fit")

        if not os.path.exists(time_directory):
            os.makedirs(time_directory)

        return time_directory
