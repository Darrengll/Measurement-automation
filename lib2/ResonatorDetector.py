from scipy import angle, argmin, linspace
from resonator_tools.circuit import notch_port, reflection_port
from numpy import abs
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


class ResonatorDetector():

    def __init__(self, frequencies=None, s_data=None, plot=True, fast=False, type = None):

        self._plot = plot
        self._fast = fast
        self._type = type
        self.set_data(frequencies, s_data)
        # self._s_data_filtered = (savgol_filter(real(self._s_data), 21, 2)\
        #                         + 1j*savgol_filter(imag(self._s_data), 21, 2))
        # self._filtered_port = notch_port(frequencies, self._s_data_filtered)


    def set_data(self, frequencies, s_data):
        self._freqs = frequencies
        self._s_data = s_data
        if self._type == 'reflection':
            self._port = reflection_port(frequencies, s_data)
        else:
            self._port = notch_port(frequencies, s_data)

    def set_plot(self, plot):
        self._plot = plot

    def detect(self, verbose = False):

        frequencies, sdata = self._freqs, self._s_data

        if not self._fast:
            result = self._fit(verbose)
        else:
            amps = abs(self._s_data)
            phas = angle(self._s_data)
            min_idx = argmin(amps)
            result = frequencies[min_idx], min(amps), phas[min_idx]

        if result is not None:
            if self._plot and not verbose:
                self._port.plotall()
            return result

    def _fit(self, verbose):
        scan_range = self._freqs[-1] - self._freqs[0]

        self._port.autofit()

        if verbose:
            self._port.plotall()

        if not self._freqs[0] < self._port.fitresults["fr"] < self._freqs[-1] \
                or self._port.fitresults["Ql"] > 20000:
            # fit failed
            return None

        min_idx = argmin(abs(self._s_data))
        expected_frequency = self._freqs[min_idx]
        expected_amplitude = abs(self._s_data)[min_idx]

        fine_freqs = linspace(self._freqs[0], self._freqs[-1], 10000)
        if self._type == 'reflection':
            fine_model = self._port._S11_directrefl(fine_freqs,
                                           fr=self._port.fitresults["fr"],
                                           Ql=self._port.fitresults["Ql"],
                                           Qc=self._port.fitresults["Qc"],
                                           a=self._port.fitresults["a"],
                                           alpha=self._port.fitresults["alpha"],
                                           delay=self._port.fitresults["delay"])

        else:
            fine_model = self._port._S21_notch(fine_freqs,
                                           fr=self._port.fitresults["fr"],
                                           Ql=self._port.fitresults["Ql"],
                                           Qc=self._port.fitresults["absQc"],
                                           phi=self._port.fitresults["phi0"],
                                           a=self._port.fitresults["a"],
                                           alpha=self._port.fitresults["alpha"],
                                           delay=self._port.fitresults["delay"])

        if verbose:
            plt.plot(fine_freqs, abs(fine_model))
        fit_min_idx = argmin(abs(fine_model))
        fit_frequency = fine_freqs[fit_min_idx]
        fit_amplitude = min(abs(fine_model))
        fit_angle = angle(fine_model)[fit_min_idx]
        res_width = fit_frequency / self._port.fitresults["Ql"]

        if abs(fit_frequency - expected_frequency) < 0.1 * res_width and \
                abs(fit_amplitude - expected_amplitude) < 5 * expected_amplitude:
                return fit_frequency, fit_amplitude, fit_angle
        else:
            return None
