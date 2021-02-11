from copy import deepcopy

from scipy import fftpack as fp
import numpy as np
from importlib import reload
from matplotlib import pyplot as plt, colorbar

from lib.iq_downconversion_calibration import IQDownconversionCalibrationResult
from lib2.IQPulseSequence import IQPulseBuilder
import lib2.waveMixing
from lib2.MeasurementResult import MeasurementResult

reload(lib2.waveMixing)
from lib2.waveMixing import PulseMixing


class StimulatedEmission(PulseMixing):

    def __init__(self, name, sample_name, comment, q_lo=None, q_iqawg=None, dig=None):
        super().__init__(name, sample_name, comment, q_lo=q_lo, q_iqawg=q_iqawg, dig=dig)
        self._delay_correction = 0
        self._measurement_result = StimulatedEmissionResult(name, sample_name)
        self._sequence_generator = IQPulseBuilder.build_stimulated_emission_sequence
        self._only_second_pulse_trace = None
        self._subtract_pi = False
        self._down_conversion_calibration = None
        self.data_backup = []

    def set_fixed_parameters(self, pulse_sequence_parameters, freq_limits=(-50e6, 50e6), delay_correction=0,
                             down_conversion_calibration=None, subtract_pi=False,
                             q_lo_params=None, q_iqawg_params=None, dig_params=None):
        """

        Parameters
        ----------
        pulse_sequence_parameters: dict
            single pulse parameters
        freq_limits: tuple of 2 values
        delay_correction: int
         A correction of a digitizer delay given in samples
         For flexibility, it is applied after the measurement. For example, when you look at the trace and decide, that
         the digitizer delay should have been different
        q_lo_params
        q_iqawg_params
        dig_params

        Returns
        -------
        Nothing
        """
        q_lo_params[0]["power"] = q_iqawg_params[0]["calibration"].get_radiation_parameters()["lo_power"]
        # used as a snapshot of initial seq pars structure passed into measurement
        self._pulse_sequence_parameters_init = deepcopy(pulse_sequence_parameters)
        super().set_fixed_parameters(pulse_sequence_parameters,
                                     freq_limits=freq_limits,
                                     q_lo_params=q_lo_params,
                                     q_iqawg_params=q_iqawg_params,
                                     dig_params=dig_params)

        self._down_conversion_calibration = down_conversion_calibration
        self._subtract_pi = subtract_pi
        self._delay_correction = delay_correction

        meas_data = self._measurement_result.get_data()
        # frequency is already set in call of 'super()' class method
        meas_data["frequency"] = self._frequencies
        meas_data["delay_correction"] = self._delay_correction
        meas_data["nfft"] = self._nfft
        meas_data["start_idx"] = self._start_idx
        meas_data["end_idx"] = self._end_idx
        meas_data["time"] = np.linspace(0, self._nfft*self._dig[0].get_sample_rate()*1e9, self._nfft, endpoint=False)
        self._measurement_result.set_data(meas_data)

    def sweep_pulse_amplitude(self, amplitude_coefficients):
        self._name += "_ampl"
        swept_pars = {"Pulse amplitude coefficient": (self._set_excitation_amplitude, amplitude_coefficients)}
        self.set_swept_parameters(**swept_pars)

    def _set_excitation_amplitude(self, amplitude_coefficient):
        self._pulse_sequence_parameters["excitation_amplitudes"] = [amplitude_coefficient]
        self._output_pulse_sequence()

    def _measure_one_trace(self):
        data = super()._measure_one_trace()
        N = len(data)
        time = np.linspace(0, N / self._dig[0].get_sample_rate() * 1e9, N)
        self.dataI.append(data[0::2])
        self.dataQ.append(data[1::2])
        if self._down_conversion_calibration is not None:
            time_cal, data = IQDownconversionCalibrationResult.\
                apply_calibration(time, data,
                                  self._down_conversion_calibration)
            if not "time_cal" in self._measurement_result.get_data():
                tmp = self._measurement_result.get_data()
                shift = self._delay_correction
                tmp["time_cal"] = time_cal[shift:self._nfft + shift]
                self._measurement_result.set_data(tmp)
        return data

    def _recording_iteration(self):
        data = self._measure_one_trace()
        if self._subtract_pi:
            phase = 0
            if "phase_shifts" in self._pulse_sequence_parameters.keys():
                phase = self._pulse_sequence_parameters["phase_shifts"][0]
            self._pulse_sequence_parameters["phase_shifts"] = [phase + np.pi]
            data = data - self._measure_one_trace()
            self._pulse_sequence_parameters["phase_shifts"] = [phase]
        return data


class StimulatedEmissionResult(MeasurementResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._if_freq = None
        self.ylabel = r"$\vert F[ \vert V(t) \vert**2 ](f)\vert $, dBm"
        self._parameter_name = None

    def func_over_trace(self,trace):
        # could be np.real, np.imag, etc.
        return np.abs(trace)**2

    def log_func(self, yf):
        # could be 20 * np.log10(yf), or just yf
        return 10 * np.log10(yf)

    def set_parameter_name(self, parameter_name):
        self._parameter_name = parameter_name

    def _prepare_figure(self):
        self._last_tr = None
        self._peaks_last_tr = None
        fig = plt.figure(figsize=(16, 9))

        # fourier data axis
        ax_fft = fig.add_subplot(211)
        ax_fft.ticklabel_format(axis='x', style='plain', scilimits=(-2, 2))
        ax_fft.set_ylabel(self.ylabel)
        ax_fft.set_xlabel("Frequency, Hz")
        ax_fft.grid()

        # time domain axis
        ax_td = fig.add_subplot(212)
        ax_td.set_ylabel(r"$\left<V(t)\right>$, mV")
        ax_td.set_xlabel("t, ns")
        ax_td.grid()

        fig.tight_layout()
        return fig, (ax_fft, ax_td), (None,)

    def _plot(self, data):
        ax_fft, ax_td = self._axes
        if "data" not in data.keys():
            return

        t, y, freqs, yf, colors, legend = self._prepare_data_for_plot(data)

        ax_fft.cla()
        ax_td.cla()
        for i in range(self._iter):
            ax_fft.plot(freqs, yf[i], color=colors[i])
            ax_td.plot(t, y[i], color=colors[i])
        ax_fft.legend(legend, title=self._parameter_name)
        ax_fft.grid()
        ax_fft.relim()
        ax_fft.autoscale_view(True, True, True)

        ax_td.grid()
        ax_td.legend(legend, title=self._parameter_name)
        ax_td.relim()
        ax_td.autoscale_view(True, True, True)

    def _prepare_data_for_plot(self, data):
        amps = data[self._parameter_names[0]]
        freqs = data["frequency"]
        nfft = data["nfft"]
        shift = data["delay_correction"]  # number of samples to shift the trace
        traces = data["data"][:, shift:nfft+shift]
        self._iter = len(traces[traces[:, 0] != 0, 0])
        traces = traces[:self._iter]
        y = self.func_over_trace(traces)
        yf = self.log_func(np.abs(fp.fftshift(fp.fft(y, n=nfft)) / nfft))
        start_idx = data["start_idx"]
        end_idx = data["end_idx"]
        N = len(amps)
        t = data["time_cal"]
        colors = plt.cm.viridis_r(np.linspace(0, 1, N))
        legend = [f"{amps[i]:.2f}" for i in range(self._iter)]
        return t, y, freqs, yf[:, start_idx:end_idx+1], colors, legend
