from scipy import *
from resonator_tools.circuit import notch_port, reflection_port
from numpy import abs, pi, exp, angle, unwrap, arctan
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from lib2.GlobalParameters import *
from scipy.optimize import curve_fit

class ResonatorDetector():

    def __init__(self, frequencies=None, s_data=None, plot=True, fast=True, type = None):
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

    def detect(self):

        frequencies, sdata = self._freqs, self._s_data

        if not self._fast:
            result = self._fit()
        else:
            if GlobalParameters().resonator_types['transmission'] == True:
                amps = abs(self._s_data)
                phas = angle(self._s_data)
                min_idx = argmin(amps)
                result = frequencies[min_idx], min(amps), phas[min_idx]
            else:
                amps = abs(self._s_data)
                phas = angle(self._s_data)

                unwrapped_phase = unwrap(phas)
                max_idx = argmin(diff(unwrapped_phase))
                fr = self._freqs[max_idx]

                def Func(x,theta0, Ql, fr, slope):
                       return theta0+2.*arctan(2.*Ql*(1.-x/fr))-slope*x

                p0 = (0, 900, fr, 0)
                p_opt, cov = curve_fit(Func,self._freqs, unwrap(angle(self._s_data)), p0 = p0)


                '''
                filter_window = len(phas) // 10
                if filter_window % 2 == 0:
                    filter_window += 1
                filter_polyorder = 3
                filtered_uphase = savgol_filter(unwrapped_phase, filter_window, filter_polyorder)
                delay = abs(diff(filtered_uphase))

                max_idx = argmax(delay)
                result = frequencies[max_idx], amps[max_idx], phas[max_idx]
                '''
                phase = Func(p_opt[2], p_opt[0], p_opt[1], p_opt[2], p_opt[3])
                amp = amps[max_idx]
                result = p_opt[2], amp, phase

        if result is not None:
            if self._plot and not self._fast:
                self._port.plotall()
            return result

    def _fit(self):


        scan_range = self._freqs[-1] - self._freqs[0]

        self._port.autofit()

        if not self._freqs[0] < self._port.fitresults["fr"] < self._freqs[-1] \
                or self._port.fitresults["Ql"] > 20000:
            # fit failed
            return None

        min_idx = argmin(abs(self._s_data))
        expected_frequency = self._freqs[min_idx]
        expected_amplitude = abs(self._s_data)[min_idx]

        fit_min_idx = argmin(abs(self._port.z_data_sim))

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

        #plt.plot(fine_freqs, abs(fine_model))
        fit_frequency = fine_freqs[argmin(abs(fine_model))]
        fit_amplitude = min(abs(self._port.z_data_sim))
        fit_angle = angle(self._port.z_data_sim)[fit_min_idx]
        res_width = fit_frequency / self._port.fitresults["Ql"]

        if abs(fit_frequency - expected_frequency) < 0.1 * res_width and \
                abs(fit_amplitude - expected_amplitude) < 5 * expected_amplitude:
                return fit_frequency, fit_amplitude, fit_angle
        else:
            return None

'''

from scipy import *
from resonator_tools.circuit import notch_port, reflection_port
from numpy import abs, pi, exp
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from lib2.GlobalParameters import *


class ResonatorDetector():

    def __init__(self, frequencies=None, s_data=None, plot=True, fast=True, type = None):
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

    def detect(self):

        frequencies, sdata = self._freqs, self._s_data
        if not self._fast:
            result = self._fit()
        else:
            if GlobalParameters().resonator_types['transmission'] == True:
                amps = abs(self._s_data)
                phas = angle(self._s_data)
                min_idx = argmin(amps)
                result = frequencies[min_idx], min(amps), phas[min_idx]
            else:
                amps = abs(self._s_data)
                phas = angle(self._s_data)
                unwrapped_phase = unwrap(phas)
                filter_window = len(phas) // 10
                if filter_window % 2 == 0:
                    filter_window += 1
                filter_polyorder = 3
                filtered_uphase = savgol_filter(unwrapped_phase, filter_window, filter_polyorder)
                delay = abs(diff(filtered_uphase))

                max_idx = argmax(delay)
                result = frequencies[max_idx], amps[max_idx], phas[max_idx]
                return result

        if result is not None and not self._fast:
            if self._plot:
                self._port.plotall()
            return result

    def _fit(self):


        scan_range = self._freqs[-1] - self._freqs[0]

        self._port.autofit()

        if not self._freqs[0] < self._port.fitresults["fr"] < self._freqs[-1] \
                or self._port.fitresults["Ql"] > 20000:
            # fit failed
            return None

        min_idx = argmin(abs(self._s_data))
        expected_frequency = self._freqs[min_idx]
        expected_amplitude = abs(self._s_data)[min_idx]

        fit_min_idx = argmin(abs(self._port.z_data_sim))

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

        #plt.plot(fine_freqs, abs(fine_model))
        fit_frequency = fine_freqs[argmin(abs(fine_model))]
        fit_amplitude = min(abs(self._port.z_data_sim))
        fit_angle = angle(self._ports.z_data_sim)[fit_min_idx]
        res_width = fit_frequency / self._port.fitresults["Ql"]

        if abs(fit_frequency - expected_frequency) < 0.1 * res_width and \
                abs(fit_amplitude - expected_amplitude) < 5 * expected_amplitude:
                return fit_frequency, fit_amplitude, fit_angle
        else:
            return None
'''