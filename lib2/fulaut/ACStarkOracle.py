from numpy import mean, median, greater, array, concatenate, argmin
from scipy.optimize import curve_fit, bisect
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt

class ACStarkOracle:


    def __init__(self, ac_stark_result, chi = 1e-3, N_photons = 100, plot = False):

        self._result = ac_stark_result
        self._plot = plot
        self._chi = chi
        self._N_photons = N_photons
        self._f_max, self._k, self._cov = None, None, None

    def launch(self):

        self._extract_data()
        self._fit_spectral_line()
        self._fit_exponent()
        if self._plot:
            self._plot_result()

        power = bisect(lambda power: (-self._exponent(power, self._k, self._f_max)+self._f_max)
                                      -self._N_photons*self._chi, -70, -10)

        return self._f_max*1e9, power

    def _extract_data(self):
        data = self._result.get_data()

        self._f_res = self._result.get_context().get_equipment()["vna"][0]["freq_limits"][0]
        self._frequencies = data["Frequency [Hz]"]
        self._powers = data["Readout power [dBm]"]
        self._s_data = (data["data"].T - mean(data["data"], -1)).T

    def _fit_spectral_line(self):

        first_row = abs(self._s_data)[0]
        highest_peak = argrelextrema(first_row, greater, order=100)[0][-1]

        bounds = ((min(first_row), min(first_row), min(self._frequencies), 5e6),
                  (max(first_row), max(first_row), max(self._frequencies), 20e6))

        p_opt, p_cov = curve_fit(self._lorentzian_peak, self._frequencies, first_row,
                                 p0=(max(first_row), median(first_row), self._frequencies[highest_peak], 10e6),
                                 bounds=bounds)

        q_freqs = [p_opt[-2]]
        for row in abs(self._s_data)[1:]:
            bounds = ((min(row), min(row), min(self._frequencies), 1e6),
                      (max(row), max(row), max(self._frequencies), 20e6))
            p_opt, p_cov = curve_fit(self._lorentzian_peak, self._frequencies, row,
                                     p0=(max(row), median(row), p_opt[-2], p_opt[-1]),
                                     bounds=bounds)
            q_freqs.append(p_opt[-2])
        self._q_freqs = array(q_freqs)

    def _fit_exponent(self):
        p_opt, self._cov = curve_fit(self._exponent, self._powers,
                            self._q_freqs / 1e9, p0=(self._q_freqs[0]/1e9, -100))
        self._k, self._f_max = p_opt

    def _plot_result(self):
        fig = plt.figure()
        fig.canvas.set_window_title(self._result._name + "-fit")

        self._n_photons = abs(10 ** (self._powers / 10) * self._k / self._chi)
        plt.pcolormesh(self._n_photons, self._frequencies/1e9, abs(self._s_data - self._s_data[0,0]).T)
        plt.plot(self._n_photons, self._q_freqs / 1e9, "r.")
        plt.plot(self._n_photons, self._exponent(self._powers, self._k, self._f_max), "orange")
        plt.xlabel(r"Shift [$\chi$ = %.2f MHz]" % (self._chi*1e3))
        plt.ylabel("Frequency [GHz]")
        # plt.xscale("log")
        plt.title("AC-Stark calibration")

    def _exponent(self, power, k, f_max):
        return 10 ** (power / 10) * k + f_max

    def _lorentzian_peak(self, frequency, amplitude, offset, res_frequency, width):
        return amplitude * (0.5 * width) ** 2 / ((frequency - res_frequency) ** 2 + (0.5 * width) ** 2) + offset

