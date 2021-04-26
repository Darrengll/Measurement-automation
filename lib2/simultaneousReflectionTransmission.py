import warnings
from copy import deepcopy
from scipy import fftpack as fp
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
from lib2.IQPulseSequence import IQPulseBuilder
from lib2.MeasurementResult import MeasurementResult
import lib2.directMeasurements.digitizerTimeResolvedDirectMeasurement as dtrdm
from typing import List
from drivers.Spectrum_m4x import SPCM
from drivers.E8257D import MXG
from drivers.IQAWG import IQAWG
from drivers.Yokogawa_GS200 import Yokogawa_GS210
from lib.iq_downconversion_calibration import IQDownconversionCalibrationResult


class SimulataneousReflectionTransmission(
        dtrdm.DigitizerTimeResolvedDirectMeasurement):

    def __init__(self, name, sample_name, devs_aliases_map,
                 plot_update_interval=1, save_traces=False):
        super().__init__(name, sample_name, devs_aliases_map,
                         plot_update_interval, save_traces)

        self._delay_correction = 0
        self._sequence_generator = \
            IQPulseBuilder.build_stimulated_emission_sequence

        # if pulse edges are being subtraced,this parameter scales
        # the default pulse edge interval (before and after main pulse)
        self._pulse_edge_mult = None

        # Get absolute value of measured traces in every iteration
        self.option_abs = False

        # Rotate the phase of the pulse for subtraction of the pa
        self._subtract_pi = False

        self._shifted_traces = []  # backup list with traces

        # Coil currents. One is used for the measurement, another for
        # shifting the qubit away from resonance
        self._main_current = 0
        self._shifted_current = 0

        self._ro_cal = None
        self._ro_cals = None
        self._downconv_cals = None
        self._shifts = None

    def _init_measurement_result(self):
        self._measurement_result = ReflectionTransmissionResult(
            self._name, self._sample_name)

    def set_fixed_parameters(self, pulse_sequence_parameters,
                             freq_limits=(-50e6, 50e6), delay_correction=0,
                             down_conversion_calibration=None,
                             subtract_pi=False,
                             pulse_edge_mult=1.0,
                             q_lo_params=None,
                             q_iqawg_params=None, dig_params=None,
                             filter=True):
        """

        Parameters
        ----------
        pulse_sequence_parameters: dict
            single pulse parameters
        freq_limits: tuple of 2 values
        delay_correction: int
         A correction of a digitizer delay given in samples
         For flexibility, it is applied after the measurement. For example,
         when you look at the trace and decide, that the digitizer delay
         should have been different
        down_conversion_calibration: IQDownconversionCalibrationResult
        subtract_pi: bool
            True if you want to make the Furier spectrum clearer by
            subtracting the same trace with pulses shifted by phase pi
        q_lo_params
        q_iqawg_params
        dig_params

        Returns
        -------
        Nothing
        """
        q_lo_params[0]["power"] = q_iqawg_params[0]["calibration"] \
            .get_radiation_parameters()["lo_power"]

        # a snapshot of initial seq pars structure passed into measurement
        self._pulse_sequence_parameters_init = deepcopy(
            pulse_sequence_parameters
        )

        super().set_fixed_parameters(
            pulse_sequence_parameters,
            freq_limits=freq_limits,
            down_conversion_calibration=down_conversion_calibration,
            q_lo_params=q_lo_params,
            q_iqawg_params=q_iqawg_params,
            dig_params=dig_params
        )

        self._subtract_pi = subtract_pi
        self._pulse_edge_mult = pulse_edge_mult
        self._delay_correction = delay_correction
        self._measurement_result._freq_lims = freq_limits
        self.apply_filter = filter # Flag: apply a digital FIR filter

        # longest repetition period is initially set with data from
        # 'pulse_sequence_paramaters'
        self.max_segment_duration = \
            pulse_sequence_parameters["repetition_period"] * \
            pulse_sequence_parameters["periods_per_segment"]

        dig = self._dig[0]

        """ Supplying additional arrays to 'self._measurement_result' class """
        meas_data = self._measurement_result.get_data()
        # frequency is already set in call of 'super()' class method
        meas_data["frequency"] = self._frequencies
        meas_data["delay_correction"] = self._delay_correction
        meas_data["start_idx"] = self._start_idx
        meas_data["end_idx"] = self._end_idx
        # time in nanoseconds
        meas_data["sample_rate"] = dig.get_sample_rate()
        self._measurement_result.set_data(meas_data)


    def _get_longest_pulse_sequence_duration(self, pulse_sequence_parameters,
                                             swept_pars):
        return 0

    def sweep_pulse_amplitude(self, amplitude_coefficients):
        self._name += "_ampl"
        swept_pars = {"Pulse amplitude coefficient": (
            self._set_excitation_amplitude, amplitude_coefficients)}
        self.set_swept_parameters(**swept_pars)

    def _set_excitation_amplitude(self, amplitude_coefficient):
        self._pulse_sequence_parameters["excitation_amplitudes"] = [
            amplitude_coefficient]
        self._output_pulse_sequence()

    def sweep_repetition_period(self, periods):
        self.set_swept_parameters(**{"Period of pulses repetition, ns": (
            self._set_repetition_period, periods)})
        self.max_segment_duration = \
            np.max(periods) * \
            self._pulse_sequence_parameters["periods_per_segment"]

        self._measurement_result.set_parameter_name(
            "Period of pulses repetition, ns")

    def _set_repetition_period(self, period):
        self._pulse_sequence_parameters["repetition_period"] = period
        self._dig[0].dur_seg_ns = \
            self._pulse_sequence_parameters["periods_per_segment"] * period
        self._output_pulse_sequence()

    def sweep_pulse_phase(self, phases):
        self._name += "_phase"
        swept_pars = {"Pulse phase, radians": (self._set_phase_shift, phases)}
        self.set_swept_parameters(**swept_pars)

    def _set_phase_shift(self, phase):
        self._pulse_sequence_parameters["phase_shifts"] = [phase]
        self._output_pulse_sequence()

    def sweep_lo_shift(self, shifts, ro_cals, downconv_cals):
        self._name += "_lo"
        self._ro_cals = ro_cals
        self._downconv_cals = downconv_cals
        self._shifts = shifts
        swept_pars = {"LO shift, Hz": (self._set_lo_shift, shifts)}
        self.set_swept_parameters(**swept_pars)

    def _set_lo_shift(self, shift):
        idx = np.abs(self._shifts - shift).argmin()
        ro_cal = self._ro_cals[idx]
        self._q_lo[0].set_frequency(ro_cal.get_lo_frequency())
        self._q_iqawg[0].set_parameters({"calibration": ro_cal})
        self._down_conversion_calibration = self._downconv_cals[idx]
        if self._down_conversion_calibration is not None:
            self._down_conversion_calibration.set_shift(shift)
        self._output_pulse_sequence()

    def _obtain_shifted_trace(self):
        self._src[0].set_current(self._shifted_current)
        time, trace = self._measure_one_trace()
        self._shifted_traces.append(trace)
        self._src[0].set_current(self._main_current)

    def _recording_iteration(self):
        time, data = self._measure_one_trace()
        self.data_backup.append(data.copy())

        if self._subtract_pi:
            phase = 0
            if "phase_shifts" in self._pulse_sequence_parameters.keys():
                phase = self._pulse_sequence_parameters["phase_shifts"][0]
            self._pulse_sequence_parameters["phase_shifts"] = [phase + np.pi]
            self._output_pulse_sequence()
            time, data_pi = self._measure_one_trace()
            self.data_pi_backup.append(data_pi.copy())
            data -= data_pi
            self._pulse_sequence_parameters["phase_shifts"] = [phase]

        # Subtract the trace with a pulse, that was shifted far from
        # resonance. Can be used to subtract the pulse shape and preserve
        # quantum oscillations
        if self._subtract_shifted:
            self._obtain_shifted_trace()
            data -= self._shifted_traces[-1]

        # Parameters 'first_pulse_start' and 'last_pulse_end' are calculated
        # and stored into 'pulse_sequence_parameters' during the last call
        # to 'self._sequence_generator' function
        pulse_start = \
            self._pulse_sequence_parameters["first_pulse_start"]  # ns
        pulse_end = \
            self._pulse_sequence_parameters["last_pulse_end"]  # ns

        pulse_edge = 0
        if self._pulse_sequence_parameters["modulating_window"] == "tukey":
            pulse_edge = self._pulse_sequence_parameters["window_parameter"]
            pulse_edge *= (pulse_end - pulse_start)/2 * self._pulse_edge_mult

        target_interval = (pulse_start + pulse_edge, pulse_end - pulse_edge)

        # constructing a mask for pulses
        def belongs(t, interval):
            return (t >= interval[0]) & (t <= interval[1])

        repetition_period = self._pulse_sequence_parameters[
            "repetition_period"]  # ns
        pulses_mask = belongs(time % repetition_period, target_interval)

        # CHANGE_1 UNCOMMENT THIS
        if self.option_abs is True:
            data = np.abs(data)**2
        else:
            if_freq = self._q_iqawg[0].get_calibration().get_if_frequency()
            data = data * np.exp(-2j * np.pi * if_freq * time * 1e-9)

        # filtering
        if self.apply_filter:
            b = signal.firwin(len(data), self._freq_limits[1],
                              fs=self._dig[0].get_sample_rate())
            data = signal.convolve(data, b, "same")

        if self._cut_everything_outside_the_pulse:
            # the trace outside pulses is set to zero
            data[np.logical_not(pulses_mask)] = 0
            max_length = int(
                self.max_segment_duration * 1e-9 *  # seconds
                self._dig[0].get_sample_rate()  # Hz
            )

            # CHANGE_1 place this after 'data = np.abs(data)'
            # subtracting pulse amplitude from pulses
            # + 'data -> data[pulse_mask]'
            data[pulses_mask] -= np.mean(data[pulses_mask])

            # CHANGE_1, copy is needed to insure reference count safety
            # if copy is ommited, then "does not own it's value" exception
            # is raised
            data = data.copy()
            data.resize(max_length)  # note that np.resize() works differently

        return data


class ReflectionTransmissionResult(MeasurementResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self.ylabel = r"$\vert F[ \vert \left<V(t)\right> \vert^2 ](" \
                      r"f)\vert$, dBm"
        self._parameter_name = None
        self._fft_vertical_shift = 50  # dB
        self._trace_vertical_shift = 0.75  # mV
        self._freq_lims = [-50, 50]  # MHz
        self._yf_lims = [-80, 0]  # dBm

        # in case custom fourier transform required
        # (at user-define frequencies)
        self._custom_fft_freqs: np.ndarray = None  # custom frequencies
        # flag that indicates that custom fft
        # should be used
        self._is_custom_fft: bool = False

    def set_custom_fft_freqs(self, freqs):
        """
        Sets custom frequencies for fourier transformation.
        Not efficient but gives freedom in choosing frequencies
        hence frequency domain interval and resolution.

        Parameters
        ----------
        freqs : np.ndarray
            1D float array with desired frequencies in Hz

        Returns
        -------
        None
        """
        self._custom_fft_freqs = freqs
        self._is_custom_fft = True

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
            self.set_is_finished(True)
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

    def func_over_trace(self, trace):
        # could be np.real, np.imag, etc.
        return np.abs(trace)

    def log_func(self, yf):
        # could be 20 * np.log10(yf), or just yf
        return 20 * np.log10(yf)

    def set_parameter_name(self, parameter_name):
        self._parameter_name = parameter_name

    def _prepare_figure(self):
        self._last_tr = None
        self._peaks_last_tr = None
        fig = plt.figure(figsize=(16, 9))
        # 2x2 grid for axes to occupy several cells
        gs = fig.add_gridspec(2, 2)

        # fourier data axis
        ax_fft = fig.add_subplot(gs[0, :])  # occupies entire first row
        ax_fft.ticklabel_format(axis='x', style='plain')#, scilimits=(-2, 2))
        # ax_fft.set_ylabel(self.ylabel)
        ax_fft.set_xlabel("Frequency, MHz")
        ax_fft.grid()

        # time domain axes
        ax_real = fig.add_subplot(gs[1, 0])  # bottom left in 2x2 grid
        # ax_real.set_ylabel(r"$\left<V(t)\right>$, mV")
        ax_real.set_xlabel("t, ns")
        ax_real.grid()

        ax_imag = fig.add_subplot(gs[1, 1], sharex=ax_real, sharey=ax_real)  #
        # bottom
        # right in 2x2
        # grid
        # ax_imag.set_ylabel(r"$\left<V(t)\right>$, mV")
        ax_imag.set_xlabel("t, ns")
        ax_imag.grid()

        fig.tight_layout()
        return fig, (ax_fft, ax_real, ax_imag), (None,)

    def _plot(self, data):
        ax_fft, ax_real, ax_imag = self._axes
        if "data" not in data.keys():
            return

        t, y, freqs, yfft_db, colors, legend = self._prepare_data_for_plot(data)

        # TODO optimize: get rid of 'cla()' and add data instead of
        #  redrawing everying for every '_plot' call
        ax_fft.cla()
        ax_real.cla()
        ax_imag.cla()

        XX, YY = np.meshgrid(t, data[self._parameter_names[0]])
        ff, pp = np.meshgrid(freqs, data[self._parameter_names[0]])
        re_max = np.max(np.abs(np.real(y)))
        im_max = np.max(np.abs(np.imag(y)))

        # for i in range(len(y)):
        #     ax_fft.plot(freqs, yfft_db[i] + i * self._fft_vertical_shift,
        #                 color=colors[i])
        # ax_real.plot(t, np.real(y[i]) + i * self._trace_vertical_shift,
        #            color=colors[i])
        ax_fft.pcolormesh(ff, pp, yfft_db, cmap="inferno")
        ax_real.pcolormesh(XX, YY, np.real(y), cmap="RdBu", vmax=re_max,
                           vmin=-re_max)
        ax_imag.pcolormesh(XX, YY, np.imag(y), cmap="RdBu", vmax=im_max,
                           vmin=-im_max)
        # ax_imag.plot(t, np.imag(y[i]) + i * self._trace_vertical_shift,
        #            color=colors[i])

        # ax_fft.legend(legend, title=self._parameter_name, loc="upper right")
        ax_fft.set_title("DFFT of abs(time trace)")
        ax_fft.grid()
        ax_fft.set_xlim(self._freq_lims)
        # ax_fft.relim()
        ax_fft.set_ylabel(self._parameter_names[0])
        # ax_fft.set_ylabel(self.ylabel)
        ax_fft.set_xlabel("Frequency, MHz")
        ax_fft.autoscale_view(True, True, True)

        ax_real.grid()
        # ax_real.legend(legend, title=self._parameter_name, loc="upper right")
        ax_real.set_xlim(t[0], t[-1])
        # ax_real.set_ylabel(r"$Re \left[ \left< V(t) \right> \right]$, mV")
        ax_real.set_ylabel(self._parameter_names[0])
        ax_real.set_xlabel("t, ns")
        ax_real.autoscale_view(True, True, True)

        ax_imag.grid()
        # ax_imag.legend(legend, title=self._parameter_name, loc="upper right")
        ax_imag.set_xlim(t[0], t[-1])
        # ax_imag.set_ylabel(r"$Im \left[ \left< V(t) \right> \right]$, mV")
        ax_imag.set_xlabel("t, ns")
        ax_imag.autoscale_view(True, True, True)

    def _prepare_data_for_plot(self, data):
        """

        Parameters
        ----------
        data : dict[str, np.ndarray]
            all data acquired during measurement

        Returns
        -------
        None
        """
        amps = data[self._parameter_names[0]]
        complex_traces = data["data"]

        # 'self._current_iteration_idx' is > 0 if this function is called
        # from 'self._plot' due to the check of "data" key presence in
        # 'self.data'. If "data" key is present, then
        available_data_n = self._iter_idx_ready[-1]

        yfft_db = None  # dB of the np.abs(yfft)
        freqs = None  # fft frequencies
        if self._is_custom_fft:
            dt = 1/data["sample_rate"]  # in seconds
            yfft_db = self.log_func(
                np.abs(
                    self.custom_fourier(
                        complex_traces,#[:available_data_n + 1],
                        dt,
                        self._custom_fft_freqs
                    )
                ) / np.sqrt(complex_traces.shape[-1])
            )
            freqs = self._custom_fft_freqs
        else:
            yfft_db = self.log_func(
                np.abs(
                    np.fft.fftshift(
                        np.fft.fft(
                            complex_traces,#[:available_data_n+1],
                            axis=-1
                        ),
                        axes=-1
                    ) / np.sqrt(complex_traces.shape[-1])
                )
            )

            # 'sample_rate' in seconds so 'freqs' in Hz
            freqs = fp.fftshift(
                fp.fftfreq(complex_traces.shape[-1], d=1 / data["sample_rate"])
            )
        yfft_db[np.isneginf(yfft_db)] = 0
        # exclude singularity in logarithmic scale
        # that arises due to trace's dc offset equals zero
        dc_freq_idx = np.argmin(np.abs(freqs))
        for i in range(len(yfft_db)):
            yfft_db[i, dc_freq_idx] = np.mean(yfft_db[i])

        N = len(amps)

        colors = plt.cm.viridis_r(np.linspace(0, 1, N))
        legend = [f"{amps[i]:.2f}" for i in range(available_data_n+1)]

        t = np.linspace(
            0,
            complex_traces.shape[-1] / data["sample_rate"],
            complex_traces.shape[-1],
            endpoint=False
        ) * 1e9  # ns

        return (t, complex_traces, freqs,
                yfft_db, colors, legend)
