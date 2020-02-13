from scipy import fftpack
import numpy as np
from importlib import reload

from lib2.IQPulseSequence import IQPulseBuilder
import lib2.waveMixing

reload(lib2.waveMixing)
from lib2.waveMixing import PulseMixing


class StimulatedEmission(PulseMixing):

    def __init__(self, name, sample_name, comment, q_lo=None, q_iqawg=None, dig=None):
        super().__init__(name, sample_name, comment, q_lo=q_lo, q_iqawg=q_iqawg, dig=dig)
        self._sequence_generator = IQPulseBuilder.build_stimulated_emission_sequence
        self._only_second_pulse_trace = None
        self.data_backup = []

    def sweep_first_pulse_amplitude(self, amplitude_coefficients):
        self._name += "_ampl1"
        swept_pars = {"Pulse 1 amplitude coefficient": (self._set_excitation1_amplitude, amplitude_coefficients)}
        self.set_swept_parameters(**swept_pars)

    def sweep_second_pulse_amplitude(self, amplitude_coefficients):
        self._name += "_ampl2"
        swept_pars = {"Pulse 2 amplitude coefficient": (self._set_excitation_2_amplitude, amplitude_coefficients)}
        self.set_swept_parameters(**swept_pars)

    def _set_excitation_2_amplitude(self, amplitude_coefficient):
        super()._set_excitation2_amplitude(amplitude_coefficient)
        self._output_pulse_sequence()

    def sweep_first_pulse_phase_n_amplitude(self, phases, amplitude1_coefficients):
        self._name += "_phase1_amp_1"
        swept_pars = {"Pulse 1 phase, radians": (self._set_excitation1_phase, phases),
                      "Pulse 1 amplitude coefficient": (self._set_excitation1_amplitude, amplitude1_coefficients)}
        self.set_swept_parameters(**swept_pars)

    def _set_excitation1_phase(self, phase):
        self._pulse_sequence_parameters["phase_shifts"][0] = phase
        # self._output_pulse_sequence()

    def _subtract_second_pulse(self, data_trace):
        if self._only_second_pulse_trace is None:
            self._set_excitation1_amplitude(0)
            self._only_second_pulse_trace = self._measure_one_trace()
        return data_trace - self._only_second_pulse_trace

    def _get_second_pulse_power_density(self):
        # if self._only_second_pulse_trace is None:
        self._set_excitation1_amplitude(0)
        self._only_second_pulse_trace = self._measure_one_trace()
        self._only_second_pulse_pd = np.abs(fftpack.fftshift(fftpack.fft(self._only_second_pulse_trace, self._nfft))
                                            / self._nfft) ** 2
        return self._only_second_pulse_pd

    def _recording_iteration(self):
        data = self._measure_one_trace()
        # self.data_backup.append(data)
        fft_data = fftpack.fftshift(fftpack.fft(data, self._nfft)) / self._nfft
        # power_density = np.abs(fftpack.fftshift(fftpack.fft(data, self._nfft)) / self._nfft) ** 2
        # for debug purposes
        # self.data_backup.append(data)
        # power_density_0 = self._get_second_pulse_power_density()
        # pd = power_density - power_density_0
        self._measurement_result._iter += 1
        return fft_data[self._start_idx:self._end_idx + 1]
