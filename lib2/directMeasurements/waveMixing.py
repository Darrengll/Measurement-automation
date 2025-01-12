# import generic python packages
from importlib import reload
import copy
from collections import OrderedDict

# import scientific packages
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt, colorbar

# import drivers
from drivers.Spectrum_m4x import SPCM

# import `Measurement-automation` instruments
from lib2.Measurement import Measurement
import lib2.IQPulseSequence
reload(lib2.IQPulseSequence)
from lib2.IQPulseSequence import IQPulseBuilder
from lib2.MeasurementResult import MeasurementResult

# import `Measurement-automation` measurement base classes
from . import digitizerTimeResolvedDirectMeasurement
reload(digitizerTimeResolvedDirectMeasurement)
from .digitizerTimeResolvedDirectMeasurement \
    import DigitizerTimeResolvedDirectMeasurement

# typization imports
from typing import Union, Iterable, List
from lib.iq_downconversion_calibration import IQDownconversionCalibrationResult


class PulseMixing(DigitizerTimeResolvedDirectMeasurement):
    def __init__(self, name, sample_name, comment, q_lo=None, q_iqawg=None,
                 dig=None, save_traces=False):
        """
        Parameters
        ----------
        name : str
            name of bias measurement
        sample_name : str
            name of measured sample
        comment: str
            comment for the measurement
        q_lo, q_iqawg, dig: arrays with references
            references to LO source, AWG and the digitizer
        """
        devs_aliases_map = {"q_lo": q_lo, "q_iqawg": q_iqawg, "dig": dig}
        self._dig: List[SPCM] = None
        super().__init__(name, sample_name, devs_aliases_map,
                         plot_update_interval=1, save_traces=save_traces)
        self._measurement_result.get_context()._comment = comment
        self._sequence_generator = IQPulseBuilder.build_wave_mixing_pulses
        self._pulse_sequence_parameters = {
            "modulating_window": "rectangular",
            "excitation_amplitudes": [1],
            "z_smoothing_coefficient": 0,
        }
        # the copy for sweeping the parameters
        self._pulse_sequence_parameters_init = copy.deepcopy(
            self._pulse_sequence_parameters)

        # whether or not to nullify parts of data that
        # apriory contain only noise
        self.__cut = True
        # whether or not cut excitation pulses block
        self.__cut_pulses = False
        # how many additional space in nanoseconds to drop before and after
        # pulse boundaries
        self.__cut_padding = None

        # digitizer segment is shrinked by the following number of samples
        # to be able to catch the next trigger event
        self._pause_in_samples_before_next_trigger = 4 * 32  # > 80 samples

        # Fourier and measurement parameters
        # see purpose in 'self.set_fixed_parameters()'
        self._freq_limits = None  # tuple with if_freq limits
        self._nfft = None  # number of FFT points
        self._frequencies = None  # array of frequencies to calculate DFTT at
        # indices in self._frequencies closest to the START and END of the
        # desired frequencies interval stored into 'self._freq_limits'
        self._start_idx = None
        self._end_idx = None

    def _init_measurement_result(self):
        self._measurement_result = PulseMixingResult(self._name,
                                                     self._sample_name)

    def set_fixed_parameters(self, pulse_sequence_parameters,
                             freq_limits=(0, 50e6),
                             down_conversion_calibration=None,
                             q_lo_params=None, q_iqawg_params=None,
                             dig_params=None):
        """

        Parameters
        ----------
        pulse_sequence_parameters
        freq_limits
        down_conversion_calibration
        q_lo_params
        q_iqawg_params
        dig_params : list[dict[str,Any]]

        Returns
        -------

        """
        q_lo_params[0]["power"] = q_iqawg_params[0][
            "calibration"].get_lo_power()

        # used as a snapshot of initial seq pars structure passed into
        # measurement
        self._pulse_sequence_parameters_init.update(pulse_sequence_parameters)
        self._pulse_sequence_parameters = copy.deepcopy(
            self._pulse_sequence_parameters)

        super().set_fixed_parameters(pulse_sequence_parameters,
                                     down_conversion_calibration=
                                     down_conversion_calibration,
                                     q_lo_params=q_lo_params,
                                     q_iqawg_params=q_iqawg_params,
                                     dig_params=dig_params)

        dig = self._dig[0]
        self._pause_in_samples_before_next_trigger += \
            pulse_sequence_parameters["digitizer_delay"] * \
            dig.get_sample_rate()*1e-9

        # Fourier and measurement parameters setup
        self._freq_limits = freq_limits
        trace_len = int(
            dig.n_seg * dig_params[0]["dur_seg"]*1e-9 * dig.get_sample_rate()
        )
        self._nfft = sp.fft._helper.next_fast_len(trace_len)
        xf = sp.fft.fftshift(
            sp.fft.fftfreq(
                self._nfft,
                1 / dig.get_sample_rate()
            )
        )
        self._start_idx = np.abs(xf - self._freq_limits[0]).argmin()
        self._end_idx = np.abs(xf - self._freq_limits[1]).argmin()

        # checking indexes for consistency
        if self._end_idx < self._start_idx:
            raise ValueError("freq_limits has wrong notation")

        self._frequencies = xf[self._start_idx:self._end_idx + 1]
        meas_data = self._measurement_result.get_data()
        meas_data["if_freq"] = self._frequencies
        self._measurement_result.set_data(meas_data)

        # to provide 'set_sideband_order()' functionality
        self._measurement_result._d_freq = pulse_sequence_parameters["d_freq"]
        self._measurement_result._if_freq = \
            q_iqawg_params[0]["calibration"].get_if_frequency()

    def _get_longest_pulse_sequence_duration(self, pulse_sequence_parameters,
                                             swept_pars):
        """
        Function calculates and return the longest pulse sequence duration
        based on pulse sequence parameters provided and
        'self._sequence_generator' implementation.

        Parameters
        ----------
        pulse_sequence_parameters : dict
            Dictionary that contain pulse sequence parameters for which you
            wish to calculate the longest duration. This parameters are fixed.

        swept_pars : dict
            Sweep parameters that are needed for calculation of the
            longest sequence.

        Returns
        -------
        float
            Longest sequence duration based on pulse sequence parameters in ns.

        Notes
        ------
        This function is introduced in the context of the solution to the
        phase jumps, caused by clock incompatibility between AWG and
        digitizer. The aim is to fix distance between digitizer measurement
        window and AWG trigger that obtains digitizer.
        The last pulse ending should stay at fixed distance from trigger event
        in contrary with previous implementation, where the start of the
        first control pulse was fixed relative to trigger event.
        The previous solution forced digitizer acquisition window (which is
        placed after the pulse sequence, usually) to shift further in
        timeline following the extension of the length of the pulse sequence.
        And due to the fact that extension length does not always coincide with
        acquisition window displacement (due to difference in AWG and
        digitizer clock period) the phase jumps arise as a problem.
        The solution is that end of the last pulse stays at the same
        distance from the trigger event and pulse sequence length extends
        "back in timeline". Together with requirement that 'repetition_period"
        is dividable by both AWG and digitizer clocks this will ensure that
        phase jumps will be neglected completely.
        """
        # TODO: this is a hotfix to satisfy requirements imposed by last merge
        #  (today is 17.03.2020)
        return 0

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
        amplitude = self._pulse_sequence_parameters_init[
            "excitation_amplitude"]
        self._pulse_sequence_parameters["excitation_amplitude"] = k * amplitude
        self._output_pulse_sequence()

    def sweep_two_excitation_amplitudes(self, amplitude1_coefficients,
                                        amplitude2_coefficients):
        self._name += "ampl1_ampl2"
        swept_pars = OrderedDict(
            [
                (
                    "Pulse 2 amplitude coefficient",
                    (
                        self._set_excitation2_amplitude,
                        amplitude2_coefficients
                    )
                ),
                (
                    "Pulse 1 amplitude coefficient",
                    (
                        self._set_excitation1_amplitude,
                        amplitude1_coefficients
                    )
                )
            ]
        )

        self.set_swept_parameters(**swept_pars)
        if ("excitation_amplitudes" in self._pulse_sequence_parameters):
            del self._pulse_sequence_parameters["excitation_amplitudes"]
        self._pulse_sequence_parameters["excitation_amplitudes"] = [None] * 2

    def _set_excitation1_amplitude(self, amplitude_coefficient):
        self._pulse_sequence_parameters["excitation_amplitudes"][
            0] = amplitude_coefficient
        self._output_pulse_sequence()

    def _set_excitation2_amplitude(self, amplitude_coefficient):
        self._pulse_sequence_parameters["excitation_amplitudes"][
            1] = amplitude_coefficient

    def sweep_excitation_duration(self, durations):
        self._name += "excitation dur"
        self.set_swept_parameters(**{"Excitation duration, ns": (
        self._set_excitation_duration, durations)})
        self._measurement_result.set_parameter_name("Excitation duration, ns")

    def _set_excitation_duration(self, duration):
        self._pulse_sequence_parameters["excitation_durations"] = \
            [duration] * len(self._pulse_sequence_parameters["pulse_sequence"])
        self._output_pulse_sequence()

    def sweep_excitation_amplitude(self, amplitudes):
        self._name += "excitation ampl"
        self.set_swept_parameters(**{"Excitation amplitude, ratio": (
        self._set_excitation_amplitude, amplitudes)})
        self._measurement_result.set_parameter_name(
            "Excitation amplitude, ratio")

    def _set_excitation_amplitude(self, amplitude):
        self._pulse_sequence_parameters["excitation_amplitudes"] = \
            [amplitude] * len(
                self._pulse_sequence_parameters["pulse_sequence"])
        self._output_pulse_sequence()

    def sweep_two_excitation_durations(self, durations1, durations2):
        self._name += "dur1_dur2"
        swept_pars = {"Pulse 1 duration, ns": (
        self._set_excitation1_duration1, durations1),
                      "Pulse 2 duration, ns": (
                      self._set_excitation1_duration2, durations2)}
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
        self.set_swept_parameters(**{"Distance between pulses, ns": (
        self._set_pulse_distance, distances)})
        self._measurement_result.set_parameter_name(
            "Distance between pulses, ns")

    def _set_pulse_distance(self, distance):
        self._pulse_sequence_parameters["pulse_distances"] = \
            [distance] * len(self._pulse_sequence_parameters["pulse_sequence"])
        self._output_pulse_sequence()

    def sweep_second_pulse(self, amplitudes, durations):
        self._name += "ampl2_dur2"
        swept_pars = {"Pulse 2 amplitude, ratio": (
        self._set_excitation2_amplitude, amplitudes),
                      "Pulse 2 duration, ns": (
                      self._set_excitation1_duration2, durations)}
        self.set_swept_parameters(**swept_pars)

    def sweep_d_freq(self, dfreqs):
        self.set_swept_parameters(
            **{"$\delta f$, Hz": (self._set_pulse_distance, dfreqs)})

    def _set_dfreq(self, dfreq):
        self._pulse_sequence_parameters["d_freq"] = dfreq
        self._output_pulse_sequence()

    def sweep_single_pulse_shift(self, pulse_i, shifts):
        self.set_swept_parameters(
            **{
                f"pulse {pulse_i} shift, ns": (
                lambda x: self._set_pulse_shift(pulse_i, x), shifts)
            }
        )

    def _set_pulse_shift(self, pulse_i, shift):
        self._pulse_sequence_parameters["pulse_shifts"][pulse_i] = shift
        self._output_pulse_sequence()

    def _measure_one_trace(self):
        """
        Function starts digitizer measurement.
        Digitizer assumed already configured and waiting for start trace.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            returns pair (time, data) np arrays
            time - real-valued 1D array. If down-conversion calibration is
            applied this array will differ from np.linspace
        """
        dig = self._dig[0]
        dig_data = dig.measure()
        # construct complex valued scalar trace
        data = dig_data[0::2] + 1j * dig_data[1::2]
        '''
        In order to allow digitizer to don't miss very next 
        trigger while the acquisition window is almost equal to 
        the trigger period, acquisition window is shrank
        by 'self.__pause_in_samples_before_trigger' samples.
        In order to obtain data of the desired length the code below extends
        trace to the requested length by appending averages to each 
        corresponding segment.
        Note: adding averages to an array does not changes its average value.
        TODO: check previous note statement (someone but Shamil).
        Finally result is flattened in order to perform DFT.
        '''
        data = data.reshape(dig.n_seg, -1)
        # 'slice_stop' variable allows to avoid production of an empty list
        # by slicing operation. Empty slice is obtained if
        # 'slice_stop = -self._n_samples_to_drop_in_end' and equals zero.
        slice_stop = data.shape[-1] - self._n_samples_to_drop_in_end
        data = data[:, self._n_samples_to_drop_by_delay:slice_stop]

        # 2D array that will be set to the trace avg value
        # and appended to the end of each segment of the trace
        # scalar average is multiplied by 'np.ones()' of the appropriate 2D
        # shape so the
        avgs_to_concat = np.full(
            (
                dig.n_seg,
                int(dig.dur_seg_samples - data.shape[-1])
            ),
            np.mean(data)
        )
        data = np.hstack((data, avgs_to_concat)).ravel()

        # Applying mixer down-conversion calibration to data
        time = np.linspace(
            0,
            len(data) / dig.get_sample_rate() * 1e9,
            len(data),
            endpoint=False
        )

        if self._down_conversion_calibration is not None:
            data = self._down_conversion_calibration.apply(data)

            if "time_cal" not in self._measurement_result.get_data():
                tmp = self._measurement_result.get_data()
                shift = \
                    self._down_conversion_calibration._IQ_delay_correction
                tmp["time_cal"] = time[shift:self._nfft + shift]
                self._measurement_result.set_data(tmp)

        # for debug purposes, saving raw data, move this section wherever
        # you feel suitable for your debug
        if self._save_traces:
            self.dataIQ.append(data)
        return time, data

    def _recording_iteration(self):
        time, data = self._measure_one_trace()

        # cutting out parts of signals that do not carry any useful information
        # configured via `self.data_cutting_config(...)`
        data = self._cut_trace(data)

        fft_data = sp.fft.fftshift(
            sp.fft.fft(data, self._nfft)
        ) / self._nfft
        yf = fft_data[self._start_idx:self._end_idx + 1]
        # dt = 1/self._dig[0].get_sample_rate()
        # yf = self.custom_fourier(data, dt, self._frequencies)
        return yf

    def _cut_trace(self, trace):
        if self.__cut is True:
            # The whole pulse sequence + readout duration after is extracted
            # untouched. Parameters 'first_pulse_start' and 'last_pulse_end'
            # are calculated and stored into 'pulse_sequence_parameters'
            # during the last call to 'self._sequence_generator' function
            first_pulse_start = self._pulse_sequence_parameters[
                "first_pulse_start"]  # ns
            last_pulse_end = self._pulse_sequence_parameters[
                "last_pulse_end"]  # ns
            readout_duration = self._pulse_sequence_parameters[
                "readout_duration"]  # ns
            # if it is configured to cut out excitation pulses block
            if self.__cut_pulses:
                target_interval = (
                    last_pulse_end + self.__cut_padding,
                    last_pulse_end + readout_duration,
                )
            else:  # excitation pulses remain untouched
                target_interval = (
                    first_pulse_start,
                    last_pulse_end + readout_duration
                )

            # constructing mask for trace to be cut out
            def belongs(t, interval):
                return (t >= interval[0]) & (t <= interval[1])

            repetition_period = self._pulse_sequence_parameters[
                "repetition_period"]  # ns
            sample_duration = 1e9 / self._dig[0].get_sample_rate()
            sampling_points_times = np.r_[0:len(trace)] * sample_duration  # ns
            sampling_points_mask = belongs(
                sampling_points_times % repetition_period,
                target_interval
            )

            # the rest of the trace is equalized to the average value
            trace[np.logical_not(sampling_points_mask)] = np.mean(
                trace[sampling_points_mask])

        return trace

    def data_cutting_config(self, cut=True, cut_pulses=False, padding=2):
        """
        Sets whether or not to exclude noisy trace from the measured traces.
        Every point that is excluded (e.g. it belongs to pulses interval or
        values beyond readout interval are set to be equal to the mean value
         of that trace)
        See parameters description for more info.

        Parameters
        ----------
        cut : bool
            If True data values that do not belong to pulse interval or readout intervals
            will be set to data.mean() values.
            Pulse interval and readout values will stay untouched.

        cut_pulses : bool
            If True pulse intervals will be set to trace average as well.
            Only readout data will remain untouched.

        padding : int
            number of additional space in nanoseconds to drop
            after pulse right boundary.

        Returns
        -------
        None
        """
        # TODO: need to delete after transferring data processing into result class
        self.__cut = cut
        self.__cut_pulses = cut_pulses
        self.__cut_padding = padding

    def _output_pulse_sequence(self, zero=False):
        dig = self._dig[0]
        timedelay = self._pulse_sequence_parameters["digitizer_delay"]
        dig.calc_and_set_trigger_delay(timedelay, include_pretrigger=True)
        self._n_samples_to_drop_by_delay =\
            dig.get_how_many_samples_to_drop_in_front()
        """
            Because readout duration coincides with trigger
        period the very next trigger is missed by digitizer.
            Decreasing segment size by fixed amount
        (e.g. 64) gives enough time to digitizer to catch the very next 
        trigger event.
            Rearm before trigger is quantity you wish to take into account
        see more on rearm timings in digitizer manual
        """
        dig.calc_segment_size(
            decrease_segment_size_by=
            self._pause_in_samples_before_next_trigger
        )
        # not working somehow, but rearming time
        # equals'80 + pretrigger' samples
        # maybe due to the fact that some extra values are sampled at the end
        # of the trace in order to make 'segment_size' in samples to be
        # dividable by 32 as required by digitizer

        self._n_samples_to_drop_in_end =\
            dig.get_how_many_samples_to_drop_in_end()
        dig.setup_averaging_mode()  # update hardware parameters

        q_pbs = [q_iqawg.get_pulse_builder() for q_iqawg in self._q_iqawg]

        # TODO: 'and (self._q_z_awg[0] is not None)'  hotfix by Shamil (below)
        # I intend to declare all possible device attributes of the measurement class in it's child class definitions.
        # So hasattr(self, "_q_z_awg") is always True
        # due to the fact that I had declared this parameter and initialized it with "[None]" in RabiFromFrequencyTEST.py
        if hasattr(self, '_q_z_awg') and (self._q_z_awg[0] is not None):
            q_z_pbs = [q_z_awg.get_pulse_builder() for q_z_awg in
                       self._q_z_awg]
        else:
            q_z_pbs = [None]

        pbs = {'q_pbs': q_pbs,
               'q_z_pbs': q_z_pbs}

        if not zero:
            seqs = self._sequence_generator(self._pulse_sequence_parameters,
                                            **pbs)
            self.seqs = seqs
        else:
            # TODO: output zero sequence
            seqs = self._sequence_generator(self._pulse_sequence_parameters,
                                            **pbs)
            self.seqs = seqs

        for (seq, q_iqawg) in zip(seqs['q_seqs'], self._q_iqawg):
            # check if output trace length is dividable by awg's output
            # trigger clock period
            # TODO: The following lines are moved to the KeysightM3202A
            #  driver. Should be deleted later from here
            # if seq.get_duration() % \
            #         q_iqawg._channels[0].host_awg.trigger_clock_period != 0:
            #     raise ValueError(
            #         "AWG output duration has to be multiple of the AWG's "
            #         "trigger clock period\n"
            #         f"requested waveform duration: {seq.get_duration()} ns\n"
            #         f"trigger clock period: {q_iqawg._channels[0].host_awg.trigger_clock_period}"
            #     )
            q_iqawg.output_pulse_sequence(seq)

    def set_target_freq_2d(self, freq):
        self._measurement_result._target_freq_2D = freq

    def set_sideband_order(self, order):
        """
        Set order to visualize during measurement process.
        Takes into account positive or negative if_freq range
        in case you interchanged I and Q inputs.

        Parameters
        ----------
        order : int
            order of the sideband produced during mixing

        Returns
        -------

        """
        # control the sign in case you interchanged I and Q outputs of mixer
        if self._freq_limits[0] > 0:
            freqs_sign = 1
        else:
            freq_sign = -1
        self._measurement_result.set_sideband_order(order, freq_sign=1)

    def custom_fourier(self, complex_data, dt, custom_freqs):
        """
        Performs fourier transform on last axis of 'complex_data'

        Parameters
        ----------
        complex_data : np.ndarray
            array that has time traces stored along last dimension
        dt : float
            distance between successive points in time domain
        custom_freqs : np.ndarray
            frequencies where you wish to perform fourier transform

        Returns
        -------
        np.ndarray

        """
        time_arr = np.linspace(
            0,
            complex_data.shape[-1] * dt,
            complex_data.shape[-1],
            endpoint=False)

        """ 
            Memory usage check. 
            Large data and frequencies arrays may cause some data to be 
            dumped to hard drive. This is to be avoided
            The upper bounds for memory usage is 1 GByte.
        """
        # np.float64 assumed
        assumed_size_bytes = len(time_arr)*len(custom_freqs)*64/8
        if assumed_size_bytes / 2**30 > 1:
            self._measurement_result.set_is_finished(True)
            raise UserWarning("Custom fourier transform requires more than "
                                "1 GB of memory, execution is terminated")

        tmp = np.exp(
            -1j * 2 * np.pi * np.tensordot(custom_freqs, time_arr, axes=0)
        )  # len(freqs) x len(time) 2D array

        # tensor contraction along last axes
        # result has shape = (complex_data.shape[:-1], len(freqs))
        fourier_data = np.tensordot(
            complex_data, tmp,
            axes=([-1], [-1])
        ) / np.sqrt(complex_data.shape[-1])
        return fourier_data


class PulseMixingResult(MeasurementResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._XX = None
        self._YY = None
        self._target_freq_2D = None

        self._d_freq = None
        self._if_freq = None
        self._amps_n_phases_mode = False

        # for debug purposes
        self._dataIQ = []

    def set_sideband_order(self, order, freq_sign):
        """
        Sets target fourier if_freq that will be visualized.

        Parameters
        ----------
        order : int
            odd integer
        freq_sign : int
            '1' - positive if_freq range assumed
            '-1' - negative if_freq range assumed

        Returns
        -------
        None
        """
        self._target_freq_2D = freq_sign * (
                    self._if_freq + order * self._d_freq)

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
            raise NotImplementedError(
                "None or more than 2 swept parameters are set")

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
            raise NotImplementedError(
                "None or more than 2 swept parameters are set")

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
        vmax = np.quantile(Z[Z != -np.inf], 0.9)
        extent = [XX[0], XX[-1], YY[0], YY[-1]]
        pow_map = ax_map.imshow(Z.T, origin='lower', cmap="inferno",
                                aspect='auto', vmax=vmax,
                                vmin=vmin, extent=extent)
        cax.cla()
        plt.colorbar(pow_map, cax=cax)
        cax.tick_params(axis='y', right='off', left='on',
                        labelleft='on', labelright='off', labelsize='10')
        last_trace_data = Z[Z != -np.inf][-(
            len(data["if_freq"])):]  # [Z != -np.inf] - flattens the array
        if self._last_tr is not None:
            self._last_tr.remove()
        self._last_tr = ax_trace.plot(data["if_freq"], last_trace_data,
                                      'b').pop(0)

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
        self._last_tr = ax_trace.plot(data["if_freq"], last_trace_y,
                                      'b').pop(0)

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

        XX, YY, Z_amps, Z_phases, last_trace_y = self._prepare_amps_n_phases_for_plot2D(
            data)

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

        phase_map = ax_map_phases.imshow(Z_phases, origin='lower',
                                         cmap="twilight_r",
                                         aspect='auto', vmax=180.,
                                         vmin=-180., extent=extent)
        cax_phases.cla()
        plt.colorbar(phase_map, cax=cax_phases)
        cax_phases.tick_params(axis='y', right=False, left=True,
                               labelleft=True, labelright=False,
                               labelsize='10')

        if self._last_tr is not None:
            self._last_tr.remove()
        self._last_tr = ax_trace.plot(data["if_freq"], last_trace_y,
                                      'b').pop(0)

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

        XX, YY, Z_re, Z_im, last_trace_y = self._prepare_re_n_im_for_plot2D(
            data)

        re_nonempty = Z_re[Z_re != 0]
        im_nonempty = Z_im[Z_im != 0]
        # re_mean = np.mean(re_nonempty)
        # im_mean = np.mean(im_nonempty)
        # re_deviation = np.ptp(re_nonempty)/2
        # im_deviation = np.ptp(im_nonempty)/2
        re_max = np.max(re_nonempty)
        re_min = np.min(re_nonempty)
        im_max = np.max(im_nonempty)
        im_min = np.min(im_nonempty)
        step_X = XX[1] - XX[0]
        step_Y = YY[1] - YY[0]
        extent = [YY[0] - 1 / 2 * step_Y, YY[-1] + 1 / 2 * step_Y,
                  XX[0] - 1 / 2 * step_X, XX[-1] + 1 / 2 * step_X]
        re_map = ax_map_re.imshow(Z_re, origin='lower', cmap="RdBu_r",
                                  aspect='auto', vmax=re_max,
                                  vmin=re_min, extent=extent)
        cax_re.cla()
        plt.colorbar(re_map, cax=cax_re)
        cax_re.tick_params(axis='y', right=False, left=True,
                           labelleft=True, labelright=False, labelsize='10')

        phase_map = ax_map_im.imshow(Z_im, origin='lower', cmap="RdBu_r",
                                     aspect='auto', vmax=im_max,
                                     vmin=im_min, extent=extent)
        cax_im.cla()
        plt.colorbar(phase_map, cax=cax_im)
        cax_im.tick_params(axis='y', right=False, left=True,
                           labelleft=True, labelright=False, labelsize='10')

        if self._last_tr is not None:
            self._last_tr.remove()
        self._last_tr = ax_trace.plot(data["if_freq"], last_trace_y,
                                      'b').pop(0)

        ax_trace.set_ylim([np.min(last_trace_y), np.max(last_trace_y)])

        # ax_map.grid(False)
        ax_trace.grid(True)
        ax_trace.axis("tight")

    def _prepare_data_for_plot1D(self, data):
        # divide by zero is regularly encountered here
        # due to the fact the during the measurement process
        # data["data"] mostly contain zero values that are repetitively
        # filled with measurement results

        power_data = 20 * np.log10(
            np.abs(data["data"]) * 1e-3 / np.sqrt(50e-3))

        if self._XX is None and self._YY is None:
            self._XX, self._YY = data[self._parameter_names[0]], data[
                "if_freq"]
        return self._XX, self._YY, power_data

    def _prepare_amps_n_phases_for_plot2D(self, data):
        freqs = data["if_freq"]
        idx = np.abs(freqs - (self._target_freq_2D)).argmin()

        complex_data = data["data"][:, :, idx].transpose()
        amplitude_data = 20 * np.log10(
            np.abs(complex_data) * 1e-3 / np.sqrt(50e-3))
        phase_data = np.angle(complex_data) / np.pi * 180
        # last nonzero data of length equal to length of the 'if_freq' array
        last_trace_y = data["data"][data["data"] != 0][
                       -len(data["if_freq"]):]
        # 1e-3 - convert mV to V
        # sqrt(50e-3) impendance 50 Ohm + convert W to mW
        last_trace_y = 20 * np.log10(
            np.abs(last_trace_y) * 1e-3 / np.sqrt(50e-3))

        if self._XX is None and self._YY is None:
            self._XX, self._YY = data[self._parameter_names[0]], data[
                self._parameter_names[1]]
        return self._XX, self._YY, amplitude_data, phase_data, last_trace_y

    def _prepare_re_n_im_for_plot2D(self, data):
        freqs = data["if_freq"]
        idx = np.abs(freqs - (self._target_freq_2D)).argmin()

        complex_data = data["data"][:, :, idx]
        idx = np.abs(complex_data).argmax()
        phi = np.angle(complex_data.item(idx))
        phi = np.pi / 2 - phi if phi > 0 else -np.pi / 2 - phi
        re_data = np.real(complex_data * np.exp(1j * 0)).transpose()
        im_data = np.imag(complex_data * np.exp(1j * 0)).transpose()
        # last nonzero data of length equal to length of the 'if_freq' array
        last_trace_y = data["data"][data["data"] != 0][
                       -len(data["if_freq"]):]
        # 1e-3 - convert mV to V
        # sqrt(50) impendance 50 Ohm
        # sqrt(1-e3) - convert sqrt(W) to sqrt(mW)
        # 20 * np.log10( sqrt(mW) ) = dBm (dBmW)
        last_trace_y = 20 * np.log10(
            np.abs(last_trace_y) * 1e-3 / np.sqrt(50) / np.sqrt(1e-3))

        if self._XX is None and self._YY is None:
            self._XX, self._YY = data[self._parameter_names[1]], data[
                self._parameter_names[0]]
        return self._XX, self._YY, re_data, im_data, last_trace_y

    def _prepare_data_for_plot2D(self, data):
        freqs = data["if_freq"]

        idx = np.abs(freqs - (self._target_freq_2D)).argmin()

        # 1e-3 - convert mV to V, sqrt(50) impendance

        amplitude_data = 20 * np.log10(
            np.abs(data["data"][:, :, idx]) * 1e-3 / np.sqrt(50e-3))
        last_trace_y = data["data"][data["data"] != 0][
                       -len(data["if_freq"]):]  # last nonzero data amount
        # 1e-3 - convert mV to V, sqrt(50) impendance
        last_trace_y = 20 * np.log10(
            np.abs(last_trace_y) * 1e-3 / np.sqrt(50e-3))

        if self._XX is None and self._YY is None:
            self._XX, self._YY = data[self._parameter_names[0]], data[
                self._parameter_names[1]]
        return self._XX, self._YY, amplitude_data, last_trace_y

    def pplot_result(self, sideband_orders=range(-9,8,1)):
        """

        Parameters
        ----------
        sideband_orders : Union[int, Iterable]
            if `int` and odd plots result for corresponding sideband order
            if Iterable, plots result for every sideband order

        Returns
        -------
        None
        """
        if isinstance(sideband_orders, int):
            sideband_orders = [sideband_orders]

        # orders in rows, re and im parts in cols
        fig, axs = plt.subplots(len(sideband_orders), 2)


class DigitizerWithPowerSweepMeasurementBase(Measurement):
    """
    Class for measurements with a Spectrum digitizer and power sweep

    This one must do:
        create Measurement object, set up all devices and take them from the class;
        set up all the parameters
        make measurements:
         -- sweep power/if_freq of one/another/both of generators
            and/or central if_freq of EXA and measure single trace / list sweep for certain frequencies
         --
    """

    def __init__(self, name, sample_name, measurement_result_class, **devs_aliases):
        """
        Parameters
        ----------
        name : str
            name of bias measurement
        sample_name : str
            name of measured sample
        measurement_result_class : type[T <= MeasurementResult]
            measurement result for appropriate data handling and visualization for thi measurement
        devs_aliases : dict[str, Any]
            same as for Measurement class

        Notes
        ---------
        vna and bias source is optional

        list_devs_names: {exa_name: default_name, src_plus_name: default_name,
                         src_minus_name: default_name, vna_name: default_name, current_name: default_name}
        """

        self._dig = devs_aliases.pop("dig", None)[0]
        super().__init__(name, sample_name, devs_aliases)
        self._devs_aliases = list(devs_aliases.keys())
        self._measurement_result = measurement_result_class(name, sample_name)

        # measurement class specific parameters section
        self._cal = None
        self._adc_parameters = None
        self._lo_parameters = None
        self._waveform_functions = {
            "CONTINUOUS TWO WAVES": self.get_two_continuous_waves,
            "CONTINUOUS WAVE": self.get_continuous_wave,
            "CONTINUOUS TWO WAVES FG": self.get_two_continuous_waves_fg
        }
        self._chosen_waveform_function = \
            self._waveform_functions["CONTINUOUS TWO WAVES"]
        self._delta = 0
        self._modulation_array = None
        self._sweep_powers = None
        self.pulse_builder = None

        self._start_idx = None
        self._end_idx = None
        self._frequencies = None
        self._freq_limits = None
        self._nfft = None

    def set_fixed_parameters(self, waveform_type, awg_parameters=(),
                             adc_parameters=(), freq_limits=(),
                             lo_parameters=()):
        """
        Parameters
        ----------
        waveform_type : str
            Choose the desired mode of operation
             One of the following is possible:
            "CONTINUOUS TWO WAVES"
            "CONTINUOUS WAVE"
            "CONTINUOUS TWO WAVES FG"

        awg_parameters : list[dict[str,Any]]
            maybe it is iqawg parameters?

        adc_parameters : list[dict[str, Any]]
            "channels" :    [1], # a list of channels to measure
            "ch_amplitude":    200, # amplitude for every channel
            "dur_seg":    100e-6, # duration of a segment in us
            "n_avg":    80000, # number of averages
            "n_seg":    2, # number of segments
            "oversampling_factor":    2, # sample_rate = max_sample_rate / oversampling_factor
            "pretrigger": 32,

        freq_limits : tuple[float]
            fourier limits for visualization
        lo_parameters : list[dict[str, Any]]

        Returns
        -------
        None

        Examples
        ________
        .ipynb

        name = "CWM_P";
        sample_name = "QOP_2_probe";
        wmBase = FourWaveMixingBase(name, sample_name, dig=[dig], lo=[exg], iqawg=[iqawg]);
        dig.stop_card()
        #awg.trigger_output_config("OFF", channel=channelI)
        #awg.trigger_output_config("ON", channel=channelQ)
        adc_pars = {"channels" :    [1], # a list of channels to measure
                    "ch_amplitude":    200, # amplitude for every channel
                    "dur_seg":    50e-6, # duration of a segment in us
                    "n_avg":    20000, # number of averages
                    "n_seg":    8, # number of segments
                    "oversampling_factor":    4, # sample_rate = max_sample_rate / oversampling_factor
                    "pretrigger": 32,
                   }
        lo_pars = { "power": lo_power,
                    "if_freq": lo_freq,
                  }

        wmBase.set_fixed_parameters(delta = 20e3, awg_parameters=[{"calibration": ro_cal}],
                                    adc_parameters=[adc_pars], freq_limits=(19.5, 20.5), lo_parameters=[lo_pars])
        wmBase.set_swept_parameters(powers_limits=(-40, 0), n_powers=201)

        #awg.trigger_output_config("ON", channel=channelQ)
        """
        self._chosen_waveform_function = self._waveform_functions[waveform_type]

        if len(awg_parameters) > 0:
            self._cal = awg_parameters[0]["calibration"]
            self._amplitudes = copy.deepcopy(self._cal._if_amplitudes)
            self.pulse_builder = WMPulseBuilder(self._cal)

        if len(adc_parameters) > 0:
            self._adc_parameters = adc_parameters[0]
            self._dig.set_oversampling_factor(self._adc_parameters["oversampling_factor"])
            self._segment_size_optimal = int(self._adc_parameters["dur_seg"] * self._dig.get_sample_rate())
            self._segment_size = self._segment_size_optimal + 32 - self._segment_size_optimal % 32
            self._bufsize = self._adc_parameters["n_seg"] * self._segment_size * 4 * len(self._adc_parameters["channels"])
            self._dig.setup_averaging_mode(self._adc_parameters["channels"], self._adc_parameters["ch_amplitude"],
                                           self._adc_parameters["n_seg"], self._segment_size,
                                           self._adc_parameters["pretrigger"],
                                           self._adc_parameters["n_avg"])

        self._freq_limits = freq_limits

        # optimal size calculation
        self._nfft = sp.fft._helper.next_fast_len(
            self._adc_parameters["n_seg"] * self._segment_size_optimal
        )

        # obtaining frequencies (frequencies is duplicating)
        xf = np.fft.fftfreq(self._nfft, 1 / self._dig.get_sample_rate()) / 1e6
        self._start_idx = np.searchsorted(
            xf[:self._nfft // 2 - 1],
            self._freq_limits[0]
        )
        self._end_idx = np.searchsorted(
            xf[:self._nfft // 2 - 1],
            self._freq_limits[1]
        )
        self._frequencies = xf[self._start_idx:self._end_idx + 1]

        self._measurement_result.get_context().update(
            {
                "calibration_results": self._cal.get_optimization_results(),
                "radiation_parameters": self._cal.get_radiation_parameters()
            }
        )

        super().set_fixed_parameters(iqawg=awg_parameters, lo=lo_parameters)
        if waveform_type == "CONTINUOUS TWO WAVES FG":
            self._iqawg[0].output_continuous_two_freq_IQ_waves(self._delta)

    def set_swept_parameters(self, powers_limits, n_powers):
        self._sweep_powers = np.linspace(*powers_limits, n_powers)
        swept_parameters = {"powers at $\\omega_{p}$": (self._set_power, self._sweep_powers)}
        super().set_swept_parameters(**swept_parameters)
        par_name = list(swept_parameters.keys())[0]
        self._measurement_result.set_parameter_name(par_name)
        # self._sources_on()

    def close_devs(self, *devs_to_close):
        if "spcm" in devs_to_close:
            self._dig.close()
        Measurement.close_devs(devs_to_close)

    def _sources_on(self):
        iq_sequence = self.pulse_builder.add_zero_pulse(10000).build()
        self._iqawg[0].output_pulse_sequence(iq_sequence)
        self._lo[0].set_output_state("ON")

    def _sources_off(self):
        iq_sequence = self.pulse_builder.add_zero_pulse(10000).build()
        self._iqawg[0].output_pulse_sequence(iq_sequence)
        self._lo[0].set_output_state("OFF")

    def srcs_power_calibration(self):
        """
        To define powers to set in setter (not implemented yet)
        """
        pass

    def _set_power(self, power):
        k = np.power(10, power / 20)
        self._chosen_waveform_function(k)
        # iq_sequence = self._chosen_waveform_function(k)
        # self._iqawg[0].output_pulse_sequence(iq_sequence)

    def get_two_continuous_waves(self, k_ampl):
        duration = 2e9 * self._adc_parameters["dur_seg"]
        return self.pulse_builder.add_simultaneous_pulses(duration, self._delta, amplitude=k_ampl).build()

    def get_two_continuous_waves_fg(self, k_ampl):
        self._iqawg[0].change_amplitudes_of_cont_IQ_waves(k_ampl)
        self._iqawg[0].update_modulation_coefficient_of_IQ_waves(2.)

    def get_continuous_wave(self, k_ampl):
        duration = 1e9 * self._adc_parameters["dur_seg"]
        return self.pulse_builder.add_sine_pulse(duration, amplitude_mult=k_ampl).build()

    def _prepare_measurement_result_data(self, parameter_names, parameters_values):
        measurement_data = super()._prepare_measurement_result_data(parameter_names, parameters_values)
        measurement_data["if_freq"] = self._frequencies
        return measurement_data

    def _recording_iteration(self):
        data = self._dig.measure(self._bufsize)  # data in mV
        # deleting extra samples from segments
        a = np.arange(self._segment_size_optimal, len(data), self._segment_size)
        b = np.concatenate([a + i for i in range(0, self._segment_size - self._segment_size_optimal)])
        data_cut = np.delete(data, b)
        yf = np.abs(np.fft.fft(data_cut, self._nfft))[self._start_idx:self._end_idx + 1] * 2 / self._nfft
        self._measurement_result._iter += 1
        return yf


class WMPulseBuilder(IQPulseBuilder):
    """IQ Pulse builder for wave mixing and for other measurements for a single qubit in line """

    def add_simultaneous_pulses(self, duration, delta_freq, phase=0, amplitude=1,
                                window="rectangular", hd_amplitude=0):
        """
        Adds two simultaneous pulses with amplitudes defined by the iqmx_calibration at frequencies
        (f_lo-f_if) ± delta_freq (or simpler w0 ± dw) and some phase to the sequence. All sine pulses will be parts
        of the same continuous wave at if_freq of f_if

        Parameters:
        -----------
        duration: float, ns
            Duration of the pulse in nanoseconds. For pulses other than rectangular
            will be interpreted as t_g (see F. Motzoi et al. PRL (2009))
        delta_freq: int, Hz
            The shift of two sidebands from the central if_freq. Ought to be > 0 Hz
        phase: float, rad
            Adds a relative phase to the outputted trace.
        amplitude: float
            Calibration if_amplitudes will be scaled by the
            amplitude_value.
        window: string
            List containing the name and the description of the modulating
            window of the pulse.
            Implemented modulations:
            "rectangular"
                Rectangular window.
            "gaussian"
                Gaussian window, see F. Motzoi et al. PRL (2009).
            "hahn"
                Hahn sin^2 window
        hd_amplitude: float
            correction for the Half Derivative method, theoretically should be 1
        """
        freq_m = self._iqmx_calibration._if_frequency - delta_freq
        freq_p = self._iqmx_calibration._if_frequency + delta_freq

        if_offsets = self._iqmx_calibration._if_offsets
        if_amplitudes = self._iqmx_calibration._if_amplitudes
        sequence_m = IQPulseBuilder(self._iqmx_calibration).add_sine_pulse(duration, phase, amplitude, window,
                                                                           hd_amplitude,
                                                                           freq_m, if_offsets/2, if_amplitudes/2).build()
        sequence_p = IQPulseBuilder(self._iqmx_calibration).add_sine_pulse(duration, phase, amplitude, window,
                                                                           hd_amplitude,
                                                                           freq_p, if_offsets/2, if_amplitudes/2).build()
        final_seq = sequence_m.direct_add(sequence_p)

        self._pulse_seq_I += final_seq._i
        self._pulse_seq_Q += final_seq._q
        return self
