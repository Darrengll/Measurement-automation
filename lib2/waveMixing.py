import warnings
from copy import deepcopy
from importlib import reload
import numpy as np
from scipy import fftpack, signal
from drivers.Spectrum_m4x import SPCM
from lib2.MeasurementResult import MeasurementResult
from matplotlib import pyplot as plt, colorbar

import lib2.IQPulseSequence

reload(lib2.IQPulseSequence)
from lib2.IQPulseSequence import IQPulseBuilder

import lib2.digitizerPulsedMeasurements.digitizerTimeResolvedDirectMeasurement

reload(lib2.digitizerPulsedMeasurements.digitizerTimeResolvedDirectMeasurement)
from lib2.digitizerPulsedMeasurements.digitizerTimeResolvedDirectMeasurement import DigitizerTimeResolvedDirectMeasurement


class PulseMixing(DigitizerTimeResolvedDirectMeasurement):
    def __init__(self, name, sample_name, comment, q_lo=None, q_iqawg=None, dig=None):
        """
        Parameters
        ----------
        name : str
            name of current measurement
        sample_name : str
            name of measured sample
        comment: str
            comment for the measurement
        q_lo, q_iqawg, dig: arrays with references
            references to LO source, AWG and the digitizer
        """
        devs_aliases_map = {"q_lo": q_lo, "q_iqawg": q_iqawg, "dig": dig}
        self._dig: list[SPCM] = None
        super().__init__(name, sample_name, devs_aliases_map, plot_update_interval=1)
        self._measurement_result = PulseMixingResult(name, sample_name)

        self._measurement_result.get_context()._comment = comment
        self._sequence_generator = IQPulseBuilder.build_wave_mixing_pulses
        self._pulse_sequence_parameters = {
            "modulating_window": "rectangular",
            "excitation_amplitudes": [1],
            "z_smoothing_coefficient": 0,
        }
        # the copy for sweeping the parameters
        self._pulse_sequence_parameters_init = deepcopy(self._pulse_sequence_parameters)

        # whether or not to nullify parts of data that apriory contain only noise
        self.__cut = True
        self.__cut_pulses = False

        # Fourier and measurement parameters
        # see purpose in 'self.set_fixed_parameters()'
        self._freq_limits = None  # tuple with frequency limits
        self._nfft = None  # number of FFT points
        self._frequencies = None
        # index in self._frequencies that is closest to the START of the desired
        # frequencies interval stored into 'self._freq_limits'
        self._start_idx = None
        # index in self._frequencies that is closest to the END of the desired
        # frequencies interval stored into 'self._freq_limits'
        self._end_idx = None

        # for debug purposes
        self.dataI = []
        self.dataQ = []

    def data_cutting_config(self, cut=True, cut_pulses=False):
        """
        Sets whether or not to exclude noisy signal from the measurement

        Parameters
        ----------
        cut : bool
            If True data values that do not belong to pulse interval or readout intervals
            will be set to data.mean() values.
            Pulse interval and readout values will stay untouched.

        cut_pulses : bool
            Considered only if cut == True.
            If True pulse intervals will be set to signal average as well.
            Only readout data will remain untouched.

        Returns
        -------
        None
        """
        # TODO: need to delete after transferring data processing into result class
        self.__cut = cut
        self.__cut_pulses = cut_pulses

    def set_fixed_parameters(self, pulse_sequence_parameters, freq_limits=(0, 50e6), q_lo_params=None,
                             q_iqawg_params=None, dig_params=None):
        q_lo_params[0]["power"] = q_iqawg_params[0]["calibration"].get_radiation_parameters()["lo_power"]

        # used as a snapshot of initial seq pars structure passed into measurement
        self._pulse_sequence_parameters_init = deepcopy(pulse_sequence_parameters)
        super().set_fixed_parameters(pulse_sequence_parameters,
                                     q_lo_params=q_lo_params,
                                     q_iqawg_params=q_iqawg_params,
                                     dig_params=dig_params)

        # Fourier and measurement parameters setup
        self._freq_limits = freq_limits
        trace_len = self._dig[0].n_seg * (self._dig[0]._segment_size - self._dig[0]._n_samples_to_drop_in_end)
        self._nfft = fftpack.helper.next_fast_len(trace_len)
        xf = fftpack.fftshift(fftpack.fftfreq(self._nfft, 1 / self._dig[0].get_sample_rate()))
        self._start_idx = np.abs(xf - self._freq_limits[0]).argmin()
        self._end_idx = np.abs(xf - self._freq_limits[1]).argmin()

        # checking indexes for consistency
        if self._end_idx < self._start_idx:
            raise ValueError("freq_limits has wrong notation")

        self._frequencies = xf[self._start_idx:self._end_idx + 1]

        # to provide 'set_sideband_order()' functionality
        self._measurement_result._d_freq = pulse_sequence_parameters["d_freq"]
        self._measurement_result._if_freq = q_iqawg_params[0]["calibration"]._if_frequency

    def sweep_power(self, powers):
        self.set_swept_parameters(**{"Powers, dB": (self._set_power, powers)})
        self._measurement_result.set_parameter_name("Powers, dB")

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
        amplitude = self._pulse_sequence_parameters_init["excitation_amplitude"]
        self._pulse_sequence_parameters["excitation_amplitude"] = k * amplitude
        self._output_pulse_sequence()

    def sweep_two_excitation_amplitudes(self, amplitude1_coefficients, amplitude2_coefficients):
        self._name += "ampl1_ampl2"
        swept_pars = {"Pulse 2 amplitude coefficient": (self._set_excitation2_amplitude, amplitude2_coefficients),
                      "Pulse 1 amplitude coefficient": (self._set_excitation1_amplitude, amplitude1_coefficients)}
        self.set_swept_parameters(**swept_pars)
        if ("excitation_amplitudes" in self._pulse_sequence_parameters):
            del self._pulse_sequence_parameters["excitation_amplitudes"]
        self._pulse_sequence_parameters["excitation_amplitudes"] = [None] * 2

    def _set_excitation1_amplitude(self, amplitude_coefficient):
        self._pulse_sequence_parameters["excitation_amplitudes"][0] = amplitude_coefficient
        self._output_pulse_sequence()

    def _set_excitation2_amplitude(self, amplitude_coefficient):
        self._pulse_sequence_parameters["excitation_amplitudes"][1] = amplitude_coefficient

    def sweep_excitation_duration(self, durations):
        self._name += "excitation dur"
        self.set_swept_parameters(**{"Excitation duration, ns": (self._set_excitation_duration, durations)})
        self._measurement_result.set_parameter_name("Excitation duration, ns")

    def _set_excitation_duration(self, duration):
        self._pulse_sequence_parameters["excitation_durations"] = \
            [duration] * len(self._pulse_sequence_parameters["pulse_sequence"])
        self._output_pulse_sequence()

    def sweep_excitation_amplitude(self, amplitudes):
        self._name += "excitation ampl"
        self.set_swept_parameters(**{"Excitation amplitude, ratio": (self._set_excitation_amplitude, amplitudes)})
        self._measurement_result.set_parameter_name("Excitation amplitude, ratio")

    def _set_excitation_amplitude(self, amplitude):
        self._pulse_sequence_parameters["excitation_amplitudes"] = \
            [amplitude] * len(self._pulse_sequence_parameters["pulse_sequence"])
        self._output_pulse_sequence()

    def sweep_two_excitation_durations(self, durations1, durations2):
        self._name += "dur1_dur2"
        swept_pars = {"Pulse 1 duration, ns": (self._set_excitation1_duration1, durations1),
                      "Pulse 2 duration, ns": (self._set_excitation1_duration2, durations2)}
        self.set_swept_parameters(**swept_pars)
        if ("excitation_duration" in self._pulse_sequence_parameters):
            del self._pulse_sequence_parameters["excitation_duration"]
        self._pulse_sequence_parameters["excitation_durations"] = [None] * 2

    def _set_excitation1_duration1(self, duration1):
        self._pulse_sequence_parameters["excitation_durations"][0] = duration1

    def _set_excitation1_duration2(self, duration2):
        self._pulse_sequence_parameters["excitation_durations"][1] = duration2
        self._output_pulse_sequence()

    def sweep_pulse_distance(self, distances):
        self.set_swept_parameters(**{"Distance between pulses, ns": (self._set_pulse_distance, distances)})
        self._measurement_result.set_parameter_name("Distance between pulses, ns")

    def _set_pulse_distance(self, distance):
        self._pulse_sequence_parameters["pulse_distances"] = \
            [distance] * len(self._pulse_sequence_parameters["pulse_sequence"])
        self._output_pulse_sequence()

    def sweep_repetition_period(self, periods):
        self.set_swept_parameters(**{"Period of pulses repetition, ns": (self._set_repetition_period, periods)})
        self._measurement_result.set_parameter_name("Period of pulses repetition, ns")

    def _set_repetition_period(self, period):
        self._pulse_sequence_parameters["repetition_period"] = period
        self._output_pulse_sequence()

    def sweep_second_pulse(self, amplitudes, durations):
        self._name += "ampl2_dur2"
        swept_pars = {"Pulse 2 amplitude, ratio": (self._set_excitation2_amplitude, amplitudes),
                      "Pulse 2 duration, ns": (self._set_excitation1_duration2, durations)}
        self.set_swept_parameters(**swept_pars)

    def sweep_d_freq(self, dfreqs):
        self.set_swept_parameters(**{"$\delta f$, Hz": (self._set_pulse_distance, dfreqs)})

    def _set_dfreq(self, dfreq):
        self._pulse_sequence_parameters["d_freq"] = dfreq
        self._output_pulse_sequence()

    def sweep_single_pulse_shift(self, pulse_i, shifts):
        self.set_swept_parameters(
            **{
                f"pulse {pulse_i} shift, ns": (lambda x: self._set_pulse_shift(pulse_i, x), shifts)
            }
        )

    def _set_pulse_shift(self, pulse_i, shift):
        self._pulse_sequence_parameters["pulse_shifts"][pulse_i] = shift
        self._output_pulse_sequence()

    def _prepare_measurement_result_data(self, parameter_names, parameters_values):
        measurement_data = super()._prepare_measurement_result_data(parameter_names, parameters_values)
        measurement_data["frequency"] = self._frequencies
        return measurement_data

    def _measure_one_trace(self):
        dig = self._dig[0]
        data = dig.measure(dig._bufsize).astype(float)

        # deleting extra samples from segments
        data_cut = SPCM.extract_useful_data(data, 2, dig._segment_size, dig.get_how_many_samples_to_drop_in_front(),
                                            dig.get_how_many_samples_to_drop_in_end())
        # convertion to mV is according to
        # https://spectrum-instrumentation.com/sites/default/files/download/m4i_m4x_22xx_manual_english.pdf
        # p.81
        data_cut = data_cut / dig.n_avg / 128 * dig.ch_amplitude
        data_cut = signal.detrend(data_cut, type='constant')

        # append 32 zero after every segment with several steps, they were deleted
        # in order to allow SPCM to receive every single trigger signal
        # this lines are reviving those missed samples and fills them with zeros.
        # First, reshaping data arrays into segments, then append 32 zeros to every segment
        # and finally flatten the result
        dataI = np.append(
            data_cut[0::2].reshape(dig.n_seg, round(data_cut.shape[0] / 2 / dig.n_seg)),
            np.zeros((dig.n_seg, 32)),
            axis=1
        ).flatten()

        dataQ = np.append(
            data_cut[1::2].reshape(dig.n_seg, round(data_cut.shape[0] / 2 / dig.n_seg)),
            np.zeros((dig.n_seg, 32)),
            axis=1
        ).flatten()

        if self.__cut is True:
            # cutting out parts of signals that do not carry any
            # useful information
            sampling_freq = self._dig[0].get_sample_rate()
            samples_n = len(dataI)
            repetition_period = self._pulse_sequence_parameters["repetition_period"]
            sampling_points_times = 1e9 * np.arange(0, samples_n / sampling_freq, 1 / sampling_freq)  # ns
            readout_duration = self._pulse_sequence_parameters["readout_duration"]  # ns

            # the whole pulse sequence + readout duration after is exctracted untouched
            # supplied during the last call to 'self._sequence_generator' function
            first_pulse_start = self._pulse_sequence_parameters["first_pulse_start"]
            # supplied during the last call to 'self._sequence_generator' function
            last_pulse_end = self._pulse_sequence_parameters["last_pulse_end"]

            if self.__cut_pulses:  # if it is configured to cut out excitation pulses
                desired_intervals = [(last_pulse_end, last_pulse_end + readout_duration)]
                # print(f"cut pulses intervals: {desired_intervals}")
            else:  # excitation pulses remain untouched
                desired_intervals = [(first_pulse_start, last_pulse_end + readout_duration)]
                # print(f"NO cut pulses intervals: {desired_intervals}")

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
                t_inside_pulse = t % repetition_period
                if belongs(t_inside_pulse, desired_intervals):
                    sampling_points_times_mask[i] = 1

            # the rest of the signal is equalized to the average value
            dataI = (dataI * sampling_points_times_mask + (1 - sampling_points_times_mask) * avgI)
            dataQ = (dataQ * sampling_points_times_mask + (1 - sampling_points_times_mask) * avgQ)
        return dataI + 1j * dataQ

    def _recording_iteration(self):
        data = self._measure_one_trace()
        # for debug purposes
        self.dataI.append(np.real(data))
        self.dataQ.append(np.imag(data))

        fft_data = fftpack.fftshift(fftpack.fft(data, self._nfft)) / self._nfft
        yf = fft_data[self._start_idx:self._end_idx + 1]
        return yf

    def _output_pulse_sequence(self, zero=False):
        dig = self._dig[0]
        timedelay = self._pulse_sequence_parameters["start_delay"] + \
                    self._pulse_sequence_parameters["digitizer_delay"]
        dig.calc_and_set_trigger_delay(timedelay, include_pretrigger=True)
        self._n_samples_to_drop_by_dig_delay = dig.get_how_many_samples_to_drop_in_front()

        dig.calc_segment_size(extra_samples_to_drop_in_end=32)  # HOTFIX FOR TRIGGER
        dig.setup_averaging_mode()

        q_pbs = [q_iqawg.get_pulse_builder() for q_iqawg in self._q_iqawg]

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

        for (seq, dev) in zip(seqs['q_seqs'], self._q_iqawg):
            dev.output_pulse_sequence(seq)
        if 'q_z_seqs' in seqs.keys():
            for (seq, dev) in zip(seqs['q_z_seqs'], self._q_z_awg):
                dev.output_pulse_sequence(seq, asynchronous=False)

    def set_target_freq_2D(self, freq):
        self._measurement_result._target_freq_2D = freq

    def set_sideband_order(self, order):
        """
        Set order to visualize during measurement process

        Parameters
        ----------
        order : int
            order of the sideband produced during mixing

        Returns
        -------

        """
        self._measurement_result.set_sideband_order(order)


class PulseMixingResult(MeasurementResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._XX = None
        self._YY = None
        self._target_freq_2D = None

        self._d_freq = None
        self._if_freq = None
        self._amps_n_phases_mode = False

    def set_sideband_order(self, order):
        """

        Parameters
        ----------
        order : int
            odd integer

        Returns
        -------

        """
        self._target_freq_2D = -(self._if_freq + order*self._d_freq)

    def set_parameter_name(self, parameter_name):
        self._parameter_name = parameter_name

    def _prepare_figure(self):
        n_parameters = len(self._parameter_names)
        if n_parameters == 1:
            return self._prepare_figure1D()
        elif n_parameters == 2:
            if self._amps_n_phases_mode:
                return self._prepare_figure2D_amps_n_phases()
            return self._prepare_figure2D_re_n_im()
        else:
            raise NotImplementedError("None or more than 2 swept parameters are set")

    def _prepare_figure1D(self):
        self._last_tr = None
        self._peaks_last_tr = None
        fig = plt.figure(figsize=(12, 8))
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
        fig = plt.figure(figsize=(12, 8))
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

    def _prepare_figure2D_amps_n_phases(self):
        self._last_tr = None
        self._peaks_last_tr = None
        fig = plt.figure(figsize=(17, 8))
        ax_trace = plt.subplot2grid((4, 2), (0, 0), colspan=2, rowspan=1)
        ax_map_amps = plt.subplot2grid((4, 2), (1, 0), colspan=1, rowspan=3)
        ax_map_phas = plt.subplot2grid((4, 2), (1, 1), colspan=1, rowspan=3)

        ax_trace.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_trace.set_xlabel("Frequency, Hz")
        ax_trace.set_ylabel("power, dB")

        ax_map_amps.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_map_amps.set_ylabel(self._parameter_names[1].upper())
        ax_map_amps.set_xlabel(self._parameter_names[0].upper())
        ax_map_amps.autoscale_view(True, True, True)
        ax_map_phas.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_map_phas.set_xlabel(self._parameter_names[0].upper())
        ax_map_phas.autoscale_view(True, True, True)
        plt.tight_layout(pad=1, h_pad=2, w_pad=-7)
        cax_amps, kw = colorbar.make_axes(ax_map_amps, aspect=40)
        cax_phas, kw = colorbar.make_axes(ax_map_phas, aspect=40)
        ax_map_amps.set_title("Amplitude, dB", position=(0.5, -0.05))
        ax_map_phas.set_title("Phase, °", position=(0.5, -0.1))
        ax_map_amps.grid(False)
        ax_map_phas.grid(False)
        fig.canvas.set_window_title(self._name)
        return fig, (ax_trace, ax_map_amps, ax_map_phas), (cax_amps, cax_phas)

    def _prepare_figure2D_re_n_im(self):
        self._last_tr = None
        self._peaks_last_tr = None
        fig = plt.figure(figsize=(17, 8))
        ax_trace = plt.subplot2grid((4, 2), (0, 0), colspan=2, rowspan=1)
        ax_map_re = plt.subplot2grid((4, 2), (1, 0), colspan=1, rowspan=3)
        ax_map_im = plt.subplot2grid((4, 2), (1, 1), colspan=1, rowspan=3)

        ax_trace.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_trace.set_xlabel("Frequency, Hz")
        ax_trace.set_ylabel("power, dB")

        ax_map_re.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_map_re.set_ylabel(self._parameter_names[1].upper())
        ax_map_re.set_xlabel(self._parameter_names[0].upper())
        ax_map_re.autoscale_view(True, True, True)
        ax_map_im.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_map_im.set_xlabel(self._parameter_names[0].upper())
        ax_map_im.autoscale_view(True, True, True)
        plt.tight_layout(pad=1, h_pad=2, w_pad=-7)
        cax_re, kw = colorbar.make_axes(ax_map_re, aspect=40)
        cax_im, kw = colorbar.make_axes(ax_map_im, aspect=40)
        ax_map_re.set_title("Real", position=(0.5, -0.05))
        ax_map_im.set_title("Imaginary", position=(0.5, -0.1))
        ax_map_re.grid(False)
        ax_map_im.grid(False)
        fig.canvas.set_window_title(self._name)
        return fig, (ax_trace, ax_map_re, ax_map_im), (cax_re, cax_im)

    def _plot(self, data):
        n_parameters = len(self._parameter_names)
        if n_parameters == 1:
            return self._plot1D(data)
        elif n_parameters == 2:
            if self._amps_n_phases_mode:
                return self._plot2D_amps_n_phases(data)
            return self._plot2D_re_n_im(data)
        else:
            raise NotImplementedError("None or more than 2 swept parameters are set")

    def _plot1D(self, data):
        ax_trace, ax_map = self._axes
        cax = self._caxes[0]
        if "data" not in data.keys():
            return

        XX, YY, Z = self._prepare_data_for_plot1D(data)

        try:
            vmax = np.max(Z[Z != -np.inf])
        except Exception as e:
            print(e)
            print(Z)
            print(Z.shape)
            print(Z[Z != -np.inf])
            print(Z[Z != -np.inf].shape)

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

        ax_map.grid(False)
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
        extent = [XX[0] - 1 / 2 * step_X, XX[-1] + 1 / 2 * step_X,
                  YY[0] - 1 / 2 * step_Y, YY[-1] + 1 / 2 * step_Y]
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

        ax_map.grid(False)
        ax_trace.grid('on')
        ax_trace.axis("tight")

    def _plot2D_amps_n_phases(self, data):
        ax_trace, ax_map_amps, ax_map_phases = self._axes
        cax_amps = self._caxes[0]
        cax_phases = self._caxes[1]
        if "data" not in data.keys():
            return

        XX, YY, Z_amps, Z_phases, last_trace_y = self._prepare_amps_n_phases_for_plot2D(data)

        amax = np.max(Z_amps[Z_amps != -np.inf])
        amin = np.quantile(Z_amps[Z_amps != -np.inf], 0.1)
        step_X = XX[1] - XX[0]
        step_Y = YY[1] - YY[0]
        extent = [XX[0] - 1 / 2 * step_X, XX[-1] + 1 / 2 * step_X,
                  YY[0] - 1 / 2 * step_Y, YY[-1] + 1 / 2 * step_Y]
        amps_map = ax_map_amps.imshow(Z_amps, origin='lower', cmap="inferno",
                                      aspect='auto', vmax=amax,
                                      vmin=amin, extent=extent)
        cax_amps.cla()
        plt.colorbar(amps_map, cax=cax_amps)
        cax_amps.tick_params(axis='y', right=False, left=True,
                             labelleft=True, labelright=False, labelsize='10')

        phase_map = ax_map_phases.imshow(Z_phases, origin='lower', cmap="twilight_r",
                                         aspect='auto', vmax=180.,
                                         vmin=-180., extent=extent)
        cax_phases.cla()
        plt.colorbar(phase_map, cax=cax_phases)
        cax_phases.tick_params(axis='y', right=False, left=True,
                               labelleft=True, labelright=False, labelsize='10')

        if self._last_tr is not None:
            self._last_tr.remove()
        self._last_tr = ax_trace.plot(data["frequency"], last_trace_y, 'b').pop(0)

        ax_trace.set_ylim([np.min(last_trace_y), np.max(last_trace_y)])

        # ax_map.grid(False)
        ax_trace.grid(True)
        ax_trace.axis("tight")

    def _plot2D_re_n_im(self, data):
        ax_trace, ax_map_re, ax_map_im = self._axes
        cax_re = self._caxes[0]
        cax_im = self._caxes[1]
        if "data" not in data.keys():
            return

        XX, YY, Z_re, Z_im, last_trace_y = self._prepare_re_n_im_for_plot2D(data)

        re_nonempty = Z_re[Z_re != 0]
        im_nonempty = Z_im[Z_im != 0]
        # re_mean = np.mean(re_nonempty)
        # im_mean = np.mean(im_nonempty)
        # re_deviation = np.ptp(re_nonempty)/2
        # im_deviation = np.ptp(im_nonempty)/2
        re_max = max(np.max(re_nonempty), -np.min(re_nonempty))
        im_max = max(np.max(im_nonempty), -np.min(re_nonempty))
        step_X = XX[1] - XX[0]
        step_Y = YY[1] - YY[0]
        extent = [YY[0] - 1 / 2 * step_Y, YY[-1] + 1 / 2 * step_Y,
                  XX[0] - 1 / 2 * step_X, XX[-1] + 1 / 2 * step_X]
        re_map = ax_map_re.imshow(Z_re, origin='lower', cmap="RdBu_r",
                                      aspect='auto', vmax=re_max,
                                      vmin=-re_max, extent=extent)
        cax_re.cla()
        plt.colorbar(re_map, cax=cax_re)
        cax_re.tick_params(axis='y', right=False, left=True,
                             labelleft=True, labelright=False, labelsize='10')

        phase_map = ax_map_im.imshow(Z_im, origin='lower', cmap="RdBu_r",
                                         aspect='auto', vmax=im_max,
                                         vmin=-im_max, extent=extent)
        cax_im.cla()
        plt.colorbar(phase_map, cax=cax_im)
        cax_im.tick_params(axis='y', right=False, left=True,
                               labelleft=True, labelright=False, labelsize='10')

        if self._last_tr is not None:
            self._last_tr.remove()
        self._last_tr = ax_trace.plot(data["frequency"], last_trace_y, 'b').pop(0)

        ax_trace.set_ylim([np.min(last_trace_y), np.max(last_trace_y)])

        # ax_map.grid(False)
        ax_trace.grid(True)
        ax_trace.axis("tight")

    def _prepare_data_for_plot1D(self, data):
        # divide by zero is regularly encountered here
        # due to the fact the during the measurement process
        # data["data"] mostly contain zero values that are repetitively
        # filled with measurement results

        power_data = 20 * np.log10(np.abs(data["data"]) * 1e-3 / np.sqrt(50e-3))

        if self._XX is None and self._YY is None:
            self._XX, self._YY = data[self._parameter_names[0]], data["frequency"]
        return self._XX, self._YY, power_data

    def _prepare_amps_n_phases_for_plot2D(self, data):
        freqs = data["frequency"]
        idx = np.abs(freqs - (self._target_freq_2D)).argmin()

        complex_data = data["data"][:, :, idx].transpose()
        amplitude_data = 20 * np.log10(np.abs(complex_data) * 1e-3 / np.sqrt(50e-3))
        phase_data = np.angle(complex_data) / np.pi * 180
        # last nonzero data of length equal to length of the 'frequency' array
        last_trace_y = data["data"][data["data"] != 0][-len(data["frequency"]):]
        # 1e-3 - convert mV to V
        # sqrt(50e-3) impendance 50 Ohm + convert W to mW
        last_trace_y = 20 * np.log10(np.abs(last_trace_y) * 1e-3 / np.sqrt(50e-3))

        if self._XX is None and self._YY is None:
            self._XX, self._YY = data[self._parameter_names[0]], data[self._parameter_names[1]]
        return self._XX, self._YY, amplitude_data, phase_data, last_trace_y

    def _prepare_re_n_im_for_plot2D(self, data):
        freqs = data["frequency"]
        idx = np.abs(freqs - (self._target_freq_2D)).argmin()

        complex_data = data["data"][:, :, idx]
        idx = np.abs(complex_data).argmax()
        phi = np.angle(complex_data.item(idx))
        phi = np.pi/2 - phi if phi > 0 else -np.pi/2 - phi
        re_data = np.real(complex_data*np.exp(1j*0)).transpose()
        im_data = np.imag(complex_data*np.exp(1j*0)).transpose()
        # last nonzero data of length equal to length of the 'frequency' array
        last_trace_y = data["data"][data["data"] != 0][-len(data["frequency"]):]
        # 1e-3 - convert mV to V
        # sqrt(50e-3) impendance 50 Ohm + convert W to mW
        last_trace_y = 20 * np.log10(np.abs(last_trace_y) * 1e-3 / np.sqrt(50e-3))

        if self._XX is None and self._YY is None:
            self._XX, self._YY = data[self._parameter_names[1]], data[self._parameter_names[0]]
        return self._XX, self._YY, re_data, im_data, last_trace_y

    def _prepare_data_for_plot2D(self, data):
        freqs = data["frequency"]

        idx = np.abs(freqs - (self._target_freq_2D)).argmin()

        # 1e-3 - convert mV to V, sqrt(50) impendance

        amplitude_data = 20 * np.log10(np.abs(data["data"][:, :, idx]) * 1e-3 / np.sqrt(50e-3))
        last_trace_y = data["data"][data["data"] != 0][-len(data["frequency"]):]  # last nonzero data amount
        # 1e-3 - convert mV to V, sqrt(50) impendance
        last_trace_y = 20 * np.log10(np.abs(last_trace_y) * 1e-3 / np.sqrt(50e-3))

        if self._XX is None and self._YY is None:
            self._XX, self._YY = data[self._parameter_names[0]], data[self._parameter_names[1]]
        return self._XX, self._YY, amplitude_data, last_trace_y
