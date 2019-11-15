from copy import deepcopy
from importlib import reload

import numpy as np
from scipy import fftpack

import lib2.IQPulseSequence
from drivers.Spectrum_m4x import SPCM
from lib2.Measurement import Measurement
from lib2.MeasurementResult import MeasurementResult

reload(lib2.IQPulseSequence)
from lib2.IQPulseSequence import IQPulseBuilder

from ..FourWaveMixingBase import FourWaveMixingResult

from typing import Union

class PulseMixingDigitizer(Measurement):
    def __init__(self, name, sample_name, comment, q_lo=None, q_awg=None, dig=None):
        """
        Parameters
        ----------
        name : str
            name of current measurement
        sample_name : str
            name of measured sample
        comment: str
            comment for the measurement
        q_lo, q_awg, dig: arrays with references
            references to LO source, AWG and the digitizer
        """
        devs_aliases = {"q_lo": q_lo, "q_awg": q_awg, "dig": dig}
        super().__init__(name, sample_name, devs_aliases, plot_update_interval=1)

        self._measurement_result = PulseMixingDigitizerResult(name, sample_name)
        self._measurement_result.get_context()._comment = comment
        self._sequence_generator = IQPulseBuilder.build_wave_mixing_pulses
        self._pulse_sequence_parameters = {
            "modulating_window": "rectangular",
            "excitation_amplitudes": [1],
            "z_smoothing_coefficient": 0,
        }
        # the copy for sweeping the parameters
        self._pulse_sequence_parameters_copy = deepcopy(self._pulse_sequence_parameters)

        # # measurement class specific parameters section
        # self._cal = None
        # self._adc_parameters = None
        # self._lo_parameters = None
        # self._waveform_functions = {"CONTINUOUS TWO WAVES": self.get_two_continuous_waves,
        #                             "CONTINUOUS WAVE": self.get_continuous_wave,
        #                             "CONTINUOUS TWO WAVES FG": self.get_two_continuous_waves_fg}
        # self._chosen_waveform_function = self._waveform_functions["CONTINUOUS TWO WAVES"]
        # self._delta = 0
        # self._modulation_array = None
        # self._sweep_powers = None
        # self.pulse_builder = None

        self.__cut = True

    def _cut_input_data(self, cut=True):
        """
        Sets whether or not to exclide noisy signal from the measurement

        Parameters
        ----------
        cut : bool

        Returns
        -------
        None
        """
        # TODO: need to delete after transferring data processing into result class
        self.__cut = cut

        self._measurement_result._cut = True

    def set_fixed_parameters(self, pulse_sequence_parameters, freq_limits=(0, 50e6), q_lo_params=None, q_awg_params=None, dig_params=None):
        q_lo_params[0]["power"] = q_awg_params[0]["calibration"].get_radiation_parameters()["lo_power"]
        dev_params = {"q_lo": q_lo_params,
                      "q_awg": q_awg_params,
                      "dig": dig_params}
        self._pulse_sequence_parameters.update(pulse_sequence_parameters)
        self._pulse_sequence_parameters_copy.update(pulse_sequence_parameters)

        # self._measurement_result.get_context().get_pulse_sequence_parameters().update(pulse_sequence_parameters)
        self._measurement_result._iter = 0
        super().set_fixed_parameters(**dev_params)
        self._measurement_result.get_context().update({
            "calibration_results": self._q_awg[0]._calibration.get_optimization_results(),
            "radiation_parameters": self._q_awg[0]._calibration.get_radiation_parameters(),
            "pulse_sequence_parameters": pulse_sequence_parameters
        })

        # Fourier and measurement parameters setup
        self._freq_limits = freq_limits
        trace_len = self._dig[0].n_seg * (self._dig[0]._segment_size - self._dig[0]._n_samples_to_drop_in_end)
        self._nfft = fftpack.helper.next_fast_len(trace_len)
        xf = fftpack.fftshift(fftpack.fftfreq(self._nfft, 1 / self._dig[0].get_sample_rate()))
        self._start_idx = np.searchsorted(xf, self._freq_limits[0])
        self._end_idx = np.searchsorted(xf, self._freq_limits[1])
        self._frequencies = xf[self._start_idx:self._end_idx + 1]

    def sweep_power(self, powers):
        self.set_swept_parameters(**{"Powers, dB": (self._set_power, powers)})
        self._measurement_result.set_parameter_name("Powers, dB")

    def sweep_excitation_duration(self, durations):
        self.set_swept_parameters(**{"Excitation duration, ns": (self._set_excitation_duration, durations)})
        self._measurement_result.set_parameter_name("Excitation duration, ns")

    def sweep_two_excitation_durations(self, durations1, durations2):
        swept_pars = {"Pulse1 duration, ns": (self._set_excitation1_duration1, durations1),
                      "Pulse2 duration, ns": (self._set_excitation1_duration2, durations2)}
        self.set_swept_parameters(**swept_pars)
        if ("excitation_duration" in self._pulse_sequence_parameters):
            del self._pulse_sequence_parameters["excitation_duration"]
        self._pulse_sequence_parameters["excitation_durations"] = [None]*2

    def sweep_pulse_distance(self, distances):
        self.set_swept_parameters(**{"Distance between pulses, ns": (self._set_pulse_distance, distances)})
        self._measurement_result.set_parameter_name("Distance between pulses, ns")

    def sweep_repetition_period(self, periods):
        self.set_swept_parameters(**{"Period of pulses repetition, ns": (self._set_repetition_period, periods)})
        self._measurement_result.set_parameter_name("Period of pulses repetition, ns")

    def sweep_d_freq(self, dfreqs):
        self.set_swept_parameters(**{"$\delta f$, Hz": (self._set_pulse_distance, dfreqs)})

    def sweep_single_pulse_shift(self, pulse_i, shifts):
        self.set_swept_parameters(
            **{
                f"pulse {pulse_i} shift, ns": (lambda x: self._set_pulse_shift(pulse_i, x), shifts)
            }
        )

    def _set_pulse_shift(self, pulse_i, shift):
        self._pulse_sequence_parameters["pulse_shifts"][pulse_i] = shift
        self._output_pulse_sequence()

    def _set_power(self, power):
        """

        Parameters
        ----------
        power : Union[float,int]
            power in dB relative to the calibration power
            the amplitudes of the IQ mixer will be changed accordingly

        Returns
        -------
        None
        """
        k = np.power(10, power / 20)
        amplitude = self._pulse_sequence_parameters_copy["excitation_amplitude"]
        self._pulse_sequence_parameters["excitation_amplitude"] = k * amplitude
        self._output_pulse_sequence()

    def _set_excitation_duration(self, duration):
        self._pulse_sequence_parameters["excitation_durations"] = \
            [duration] * len(self._pulse_sequence_parameters["pulse_sequence"])
        self._output_pulse_sequence()

    def _set_excitation1_duration1(self, duration1):
        self._pulse_sequence_parameters["excitation_durations"][0] = duration1

    def _set_excitation1_duration2(self, duration2):
        self._pulse_sequence_parameters["excitation_durations"][1] = duration2
        self._output_pulse_sequence()

    def _set_pulse_distance(self, distance):
        self._pulse_sequence_parameters["pulse_distances"] = \
            [distance]*len(self._pulse_sequence_parameters["pulse_sequence"])
        self._output_pulse_sequence()

    def _set_repetition_period(self, period):
        self._pulse_sequence_parameters["repetition_period"] = period
        self._output_pulse_sequence()

    def _set_dfreq(self, dfreq):
        self._pulse_sequence_parameters["d_freq"] = dfreq
        self._output_pulse_sequence()

    def _prepare_measurement_result_data(self, parameter_names, parameters_values):
        measurement_data = super()._prepare_measurement_result_data(parameter_names, parameters_values)
        measurement_data["frequency"] = self._frequencies
        return measurement_data

    def _recording_iteration(self):
        dig = self._dig[0]
        data = dig.measure(dig._bufsize)
        # deleting extra samples from segments
        data_cut = SPCM.extract_useful_data(data, 2, dig._segment_size, dig.get_how_many_samples_to_drop_in_front(),
                                            dig.get_how_many_samples_to_drop_in_end())
        data_cut = data_cut * dig.ch_amplitude / 128 / dig.n_avg
        dataI = data_cut[0::2]
        dataQ = data_cut[1::2]

        if self.__cut is True:
            # cutting out parts of signals that do not carry any
            # useful information
            sampling_freq = self._dig[0].get_sample_rate()
            samples_n = len(dataI)
            repetition_period = self._pulse_sequence_parameters["repetition_period"]
            sampling_points_times = 1e9 * np.arange(0, samples_n / sampling_freq, 1 / sampling_freq)  # ns
            readout_duration = self._pulse_sequence_parameters["readout_duration"]  # ns

            # the whole pulse sequence + readout duration after is exctracted untouched
            # supplied by sequence generator function
            first_pulse_start = self._pulse_sequence_parameters["first_pulse_start"]
            # supplied by sequence generator function
            last_pulse_end = self._pulse_sequence_parameters["last_pulse_end"]
            desired_intervals = [(first_pulse_start, last_pulse_end + readout_duration)]

            def belongs(t, intervals):
                ret = False
                for interval in intervals:
                    if (interval[0] <= t <= interval[1]):
                        ret = True
                        break
                return ret

            sampling_points_times_mask = np.zeros(len(sampling_points_times))
            avgI = np.mean(dataI)
            avgQ = np.mean(dataQ)

            for i, t in enumerate(sampling_points_times):
                t_loc = t % repetition_period
                if belongs(t_loc, desired_intervals):
                    sampling_points_times_mask[i] = 1

            # the rest of the signal is equalized to the average value
            dataI = (dataI * sampling_points_times_mask + (1 - sampling_points_times_mask) * avgI)
            dataQ = (dataQ * sampling_points_times_mask + (1 - sampling_points_times_mask) * avgQ)

        # for debug purposes
        self.dataI = dataI
        self.dataQ = dataQ

        fft_data = fftpack.fftshift(fftpack.fft(dataI + 1j * dataQ, self._nfft)) / self._nfft
        yf = np.abs(fft_data)[self._start_idx:self._end_idx + 1]
        self._measurement_result._iter += 1
        return yf

    def _output_pulse_sequence(self, zero=False):
        dig = self._dig[0]
        timedelay = self._pulse_sequence_parameters["digitizer_delay"]
        dig.calc_and_set_trigger_delay(timedelay, include_pretrigger=True)
        self._n_samples_to_drop_by_dig_delay = dig.get_how_many_samples_to_drop_in_front()

        dig.calc_and_set_segment_size(extra=self._n_samples_to_drop_by_dig_delay)
        dig.setup_averaging_mode()
        self._n_samples_to_drop_in_end = dig.get_how_many_samples_to_drop_in_end()


        q_pbs = [q_awg.get_pulse_builder() for q_awg in self._q_awg]

        # TODO: 'and (self._q_z_awg[0] is not None)'  hotfix by Shamil (below)
        # I intend to declare all possible device attributes of the measurement class in it's child class definitions.
        # So hasattr(self, "_q_z_awg") is always True
        # due to the fact that I had declared this parameter and initialized it with "[None]" in RabiFromFrequency.py
        if hasattr(self, '_q_z_awg') and (self._q_z_awg[0] is not None):
            q_z_pbs = [q_z_awg.get_pulse_builder() for q_z_awg in self._q_z_awg]
        else:
            q_z_pbs = [None]

        pbs = {'q_pbs': q_pbs,
               'q_z_pbs': q_z_pbs}

        if not zero:
            seqs = self._sequence_generator(self._pulse_sequence_parameters, **pbs)
            self.seqs = seqs
        else:
            # TODO: output zero sequence
            seqs = self._sequence_generator(self._pulse_sequence_parameters, **pbs)
            self.seqs = seqs

        for (seq, dev) in zip(seqs['q_seqs'], self._q_awg):
            dev.output_pulse_sequence(seq)
        if 'q_z_seqs' in seqs.keys():
            for (seq, dev) in zip(seqs['q_z_seqs'], self._q_z_awg):
                dev.output_pulse_sequence(seq, asynchronous=False)

    def set_target_freq_2D(self, freq):
        self._measurement_result._target_freq_2D = freq


from lib2.MeasurementResult import ContextBase, MeasurementResult
from matplotlib import pyplot as plt, colorbar


class PulseMixingDigitizerResult(MeasurementResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        # self._context = ContextBase(comment=input('Enter your comment: '))
        self._context = ContextBase()
        self._is_finished = False
        self._idx = []
        self._midx = []
        self._colors = []
        self._XX = None
        self._YY = None
        self._target_freq_2D = None
        self._delta = 0
        self._iter = 0
        self._cut = True

    def set_parameter_name(self, parameter_name):
        self._parameter_name = parameter_name

    def _prepare_figure(self):
        n_parameters = len(self._parameter_names)
        if n_parameters == 1:
            return self._prepare_figure1D()
        elif n_parameters == 2:
            return self._prepare_figure2D()
        else:
            raise NotImplementedError("None or more than 2 swept parameters are set")

    def _prepare_figure1D(self):
        self._last_tr = None
        self._peaks_last_tr = None
        fig = plt.figure(figsize=(19, 8))
        ax_trace = plt.subplot2grid((4, 1), (0, 0), colspan=1, rowspan=1)
        ax_map = plt.subplot2grid((4, 1), (1, 0), colspan=1, rowspan=3)
        plt.tight_layout()
        ax_map.ticklabel_format(axis='x', style='plain', scilimits=(-2, 2))
        ax_map.set_ylabel("Frequency, Hz")
        ax_map.set_xlabel(self._parameter_names[0])
        ax_trace.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_trace.set_xlabel("Frequency, Hz")
        ax_trace.set_ylabel("Fourier[V]")

        ax_map.autoscale_view(True, True, True)
        plt.tight_layout()

        cax, kw = colorbar.make_axes(ax_map, fraction=0.05, anchor=(0.0, 1.0))
        cax.set_title("power, dB")

        return fig, (ax_trace, ax_map), (cax,)

    def _prepare_figure2D(self):
        self._last_tr = None
        self._peaks_last_tr = None
        fig = plt.figure(figsize=(19, 8))
        ax_trace = plt.subplot2grid((4, 1), (0, 0), colspan=1, rowspan=1)
        ax_map = plt.subplot2grid((4, 1), (1, 0), colspan=1, rowspan=3)
        plt.tight_layout()

        ax_trace.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_trace.set_xlabel("Frequency, Hz")
        ax_trace.set_ylabel("power, dB")

        ax_map.ticklabel_format(axis='x', style='plain', scilimits=(-2, 2))
        ax_map.set_ylabel(self._parameter_names[0].upper())
        ax_map.set_xlabel(self._parameter_names[1].upper())
        ax_map.autoscale_view(True, True, True)

        plt.tight_layout()

        cax, kw = colorbar.make_axes(ax_map, fraction=0.05, anchor=(0.0, 1.0))
        cax.set_title("power, dB")

        return fig, (ax_trace, ax_map), (cax,)

    def _plot(self, data):
        n_parameters = len(self._parameter_names)
        if n_parameters == 1:
            return self._plot1D(data)
        elif n_parameters == 2:
            return self._plot2D(data)
        else:
            raise NotImplementedError("None or more than 2 swept parameters are set")

    def _plot1D(self, data):
        ax_trace, ax_map = self._axes
        cax = self._caxes[0]
        if "data" not in data.keys():
            return

        XX, YY, Z = self._prepare_data_for_plot1D(data)

        vmax = np.max(Z[Z != -np.inf])
        vmin = np.quantile(Z[Z != -np.inf], 0.1)
        extent = [XX[0], XX[-1], YY[0], YY[-1]]
        pow_map = ax_map.imshow(Z.T, origin='lower', cmap="inferno",
                                aspect='auto', vmax=vmax,
                                vmin=vmin, extent=extent)
        cax.cla()
        plt.colorbar(pow_map, cax=cax)
        cax.tick_params(axis='y', right='off', left='on',
                        labelleft='on', labelright='off', labelsize='10')
        last_trace_data = Z[Z != -np.inf][-(len(data["frequency"])):]  # [Z != -np.inf] - flattens the array
        if self._last_tr is not None:
            self._last_tr.remove()
        self._last_tr = ax_trace.plot(data["frequency"], last_trace_data, 'b').pop(0)

        ax_trace.set_ylim([np.min(last_trace_data), np.max(last_trace_data)])

        ax_map.grid('on')
        ax_trace.grid('on')
        ax_trace.axis("tight")

    def _plot2D(self, data):
        ax_trace, ax_map = self._axes
        cax = self._caxes[0]
        if "data" not in data.keys():
            return

        XX, YY, Z, last_trace_y = self._prepare_data_for_plot2D(data)

        vmax = np.max(Z[Z != -np.inf])
        vmin = np.quantile(Z[Z != -np.inf], 0.1)
        step_X = XX[1] - XX[0]
        step_Y = YY[1] - YY[0]
        extent = [XX[0] - 1/2*step_X, XX[-1] + 1/2*step_X,
                  YY[0] - 1/2*step_Y, YY[-1] + 1/2*step_Y]
        pow_map = ax_map.imshow(Z, origin='lower', cmap="inferno",
                                aspect='auto', vmax=vmax,
                                vmin=vmin, extent=extent)
        cax.cla()
        plt.colorbar(pow_map, cax=cax)
        cax.tick_params(axis='y', right='off', left='on',
                        labelleft='on', labelright='off', labelsize='10')

        if self._last_tr is not None:
            self._last_tr.remove()
        self._last_tr = ax_trace.plot(data["frequency"], last_trace_y, 'b').pop(0)

        ax_trace.set_ylim([np.min(last_trace_y), np.max(last_trace_y)])

        ax_map.grid('on')
        ax_trace.grid('on')
        ax_trace.axis("tight")

    def _prepare_data_for_plot1D(self, data):
        power_data = np.real(20 * np.log10(data["data"] * 1e3 / np.sqrt(50e-3)))

        if self._XX is None and self._YY is None:
            self._XX, self._YY = data[self._parameter_names[0]], data["frequency"]
        return self._XX, self._YY, power_data

    def _prepare_data_for_plot2D(self, data):
        freqs = data["frequency"]

        idx = np.abs(freqs - (self._target_freq_2D)).argmin()

        power_data = np.real(20 * np.log10(data["data"][:, :, idx] * 1e3 / np.sqrt(50e-3)))
        last_trace_y = data["data"][data["data"] != 0][-len(data["frequency"]):]  # last nonzero data amount
        last_trace_y = np.real(20 * np.log10(last_trace_y * 1e3 / np.sqrt(50e-3)))

        if self._XX is None and self._YY is None:
            self._XX, self._YY = data[self._parameter_names[0]], data[self._parameter_names[1]]
        return self._XX, self._YY, power_data, last_trace_y
