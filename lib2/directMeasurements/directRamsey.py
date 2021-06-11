import numpy as np
from typing import Union, List

from matplotlib import colorbar
from scipy import signal

from drivers.IQAWG import IQAWG
from drivers.Spectrum_m4x import SPCM
from drivers.E8257D import EXG, MXG

# DEVELOPMENT BLOCK
from importlib import reload

from lib.iq_downconversion_calibration import IQDownconversionCalibrator
from lib2.MeasurementResult import MeasurementResult
from . import digitizerTimeResolvedDirectMeasurement
reload(digitizerTimeResolvedDirectMeasurement)
from .digitizerTimeResolvedDirectMeasurement import DigitizerTimeResolvedDirectMeasurement

from .. import VNATimeResolvedDispersiveMeasurement1D
reload(VNATimeResolvedDispersiveMeasurement1D)
from ..VNATimeResolvedDispersiveMeasurement1D import VNATimeResolvedDispersiveMeasurement1DResult
from .. import IQPulseSequence
reload(IQPulseSequence)
from ..IQPulseSequence import IQPulseBuilder
from collections import OrderedDict
from datetime import datetime as dt
import matplotlib.pyplot as plt


class DirectRamseyBase(DigitizerTimeResolvedDirectMeasurement):

    def __init__(self, name, sample_name, plot_update_interval=1,
                 q_lo=[], q_iqawg=[], dig=[], src=[], save_traces=False):
        """

        Parameters
        ----------
        name : str
        sample_name : str
        plot_update_interval : float

        q_lo : List[EXG]
        q_iqawg : list[IQAWG]
        dig : list[SPCM]
        """
        devs_aliases_map = {"q_lo": q_lo,
                            "q_iqawg": q_iqawg,
                            "dig": dig,
                            "insweep_trg_subsys": src}

        super().__init__(name, sample_name, devs_aliases_map,
                         plot_update_interval, save_traces)
        self._measurement_result = None  # has to be initialized in child classes
        # initialize 'self._measurement_result' that is specific for particular child class
        self._init_measurement_result()
        self._sequence_generator = IQPulseBuilder.build_direct_ramsey_sequence
        self._pi_subtraction = False
        self._digital_filtering = False
        self._fir_cutoff = 150e6

    def _init_measurement_result(self):
        """
        Pure virtual function that allows child classes to initialize
        measurement_result attribute in a 'hook' fasion

        Returns
        -------
        None
        """
        raise NotImplementedError

    def _get_longest_pulse_sequence_duration(self, pulse_sequence_parameters, swept_pars):
        """
        Implementation of purely virtual function for 'DirectRabi' sequences measurements.
        Function calculates and return the longest pulse sequence duration based
        on pulse sequence parameters provided and 'self._sequence_generator' implementation.

        Parameters
        ----------
        pulse_sequence_parameters : dict
            Dictionary that contain pulse sequence parameters for which
            you wish to calculate the longest duration. This parameters are fixed.

        swept_pars : dict
            Sweep parameters that are needed for calculation of the
            longest sequence.

        Returns
        -------
        float
            Longest sequence duration based on pulse sequence parameters in ns.

        Notes
        ------
            This function is introduced in the context of the solution to the phase jumps, caused
        by clock incompatibility between AWG and digitizer. The aim is to fix distance between
        digitizer measurement window and AWG trigger that obtains digitizer.
            The last pulse ending should stay at fixed distance from trigger event in contrary with previous
        implementation, where the start of the first control pulse was fixed relative to trigger event.
            The previous solution forced digitizer acquisition window (which is placed after the pulse sequence, usually)
        to shift further in timeline following the extension of the length of the pulse sequence.
        And due to the fact that extension length does not always coincide with acquisition
        window displacement (due to difference in AWG and digitizer clock period) the phase jumps
        arise as a problem.
            The solution is that end of the last pulse stays at the same distance from the trigger event and
        pulse sequence length extendends "back in timeline". Together with requirement that 'repetition_period"
        is dividable by both AWG and digitizer clocks this will ensure that phase jumps will be neglected completely.
        """
        longest_excitation = self._pulse_sequence_parameters[
            "pi_half_pulse_duration"]
        return longest_excitation

    def _single_measurement(self):
        dig = self._dig[0]

        dig_data = dig.measure()  # in mV
        # construct complex valued scalar trace

        data = dig_data[0::2] + 1j * dig_data[1::2]

        # save traces for debug purposes
        if self._save_traces:
            self.dataIQ.append(data)
        '''
        In order to allow digitizer not to miss very next 
        trigger while the acquisition window is almost equal to 
        the trigger period, acquisition window is shrank
        by 'self.__pause_in_samples_before_trigger' samples.
        In order to obtain data of the desired length the code below adds
        'self.__pause_in_samples_before_trigger' trailing zeros to the end
        of each segment.
        Finally result is flattened in order to perform DFT.
        '''
        # 2D array that will be set to the trace avg value
        # and appended to the end of each segment of the trace
        # scalar average is multiplied by 'np.ones()' of the appropriate 2D
        # shape
        self._pause_in_samples_before_next_trigger = 0  # 32 * 4
        avgs_to_concat = np.full((dig.n_seg,
                                  self._pause_in_samples_before_next_trigger),
                                 np.mean(data))
        data = data.reshape(dig.n_seg, -1)
        # 'slice_stop' is needed in case when 'pm._n_samples_to_drop_in_end'
        # equals zero. It helps to avoid production of an empty list by slicing
        # operation with `arr[a, self._n_samples_to_drop_in_end]`.
        slice_stop = data.shape[-1] - self._n_samples_to_drop_in_end
        data = data[:, self._n_samples_to_drop_by_delay:slice_stop]

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

        return time, data

    def setup_pi_subtraction(self, val):
        """
        The phase of oscillations after the pulse does not depends on its
        phase.
        Subtraction of a pi-shifted trace from the measured trace conserves
        these oscillations and removes spurious signals which do not depend
        on the driving pulse's phase.
        Parameters
        ----------
        val: bool
            If true, the pi-shifted trace will be measured and subtracted
            from the original trace.
        """
        self._pi_subtraction = val

    def _measure_pi_shifted(self):
        """
        This method changes the phase of a driving pulse sequence by pi,
        measures the trace and then restores the phase back to not-shifted.
        Use the result later to add or subtract it from the original trace."""
        phase = 0
        if "phase_shifts" in self._pulse_sequence_parameters.keys():
            phase = self._pulse_sequence_parameters["phase_shifts"][0]
        self._pulse_sequence_parameters["phase_shifts"] = [phase + np.pi]
        self._output_pulse_sequence()
        time, data_pi = self._single_measurement()
        self._pulse_sequence_parameters["phase_shifts"] = [phase]
        return time, data_pi


class DirectRamseyFromDelay(DirectRamseyBase):
    """
        Rabi from pulse duration measurements
    """
    def __init__(self, name, sample_name, plot_update_interval=1,
                 q_lo=[], q_iqawg=[], dig=[], src=[], save_traces=False):
        """

        Parameters
        ----------
        name : str
        sample_name : str
        plot_update_interval : float

        q_lo : List[EXG]
        q_iqawg : list[IQAWG]
        dig : list[SPCM]
        """
        devs_aliases_map = {"q_lo": q_lo,
                            "q_iqawg": q_iqawg,
                            "dig": dig,
                            "insweep_trg_subsys": src}
        super().__init__(name, sample_name, plot_update_interval,
                         **devs_aliases_map, save_traces=save_traces)
        self._digital_filtering = None

    def _init_measurement_result(self):
        self._measurement_result = RamseyFromDelayResult(self._name,
                                                            self._sample_name)

    def sweep_ramsey_delay(self, digital_filtering=False):
        """
        This method must be called to set up the delay as the sweep parameter.

        Parameters
        ----------
        digital_filtering: bool
            If this argument is True, the FIR filter with 50 MHz cuttoff is
            applied to the measured trace. Gives cleaner results.
        """
        self._digital_filtering = digital_filtering

        delay_max = self._pulse_sequence_parameters["readout_duration"]
        delays = np.arange(
            0, delay_max,
            1/self._dig[0].get_sample_rate()*1e9
        )  # ns
        super().set_swept_parameters(**{"ramsey_delay":
                                            (self._output_pulse_sequence,
                                                         delays)})

    def _record_data(self):
        par_names = self._swept_pars_names

        start_time = self._measurement_result.get_start_datetime()

        parameters_values = [self._swept_pars[parameter_name][1]
                             for parameter_name in par_names]

        self._call_setters([parameters_values])

        # This should be implemented in child classes:
        self._raw_data = self._recording_iteration()

        # This may need to be extended in child classes:
        measurement_data = self._prepare_measurement_result_data(par_names,
                                                         parameters_values)
        self._measurement_result.set_data(measurement_data)
        self._measurement_result._iter_idx_ready = [len(parameters_values[0])]

        time_elapsed = dt.now() - start_time
        self._measurement_result.set_recording_time(time_elapsed)
        print(f"\nElapsed time: "
              f"{self._format_time_delta(time_elapsed.total_seconds())}")
        self._finalize()

    def _recording_iteration(self):
        time, data = self._single_measurement()
        if self._pi_subtraction:
            time, data_pi = self._measure_pi_shifted()
            data -= data_pi
        self._backup = data.copy()

        # centering around zero
        data -= np.mean(data)

        # moving fourier so the trace carrier is now at 0 Hz.
        if_freq = self._q_iqawg[0].get_calibration().get_if_frequency()  # Hz
        data = data * np.exp(-1j * 2 * np.pi * if_freq * time * 1e-9)

        if self._digital_filtering:
            b = signal.firwin(len(data), self._fir_cutoff,
                              fs=self._dig[0].get_sample_rate())
            data = signal.convolve(data, b, "same")
            data -= np.mean(data)

        return data


class RamseyFromDelayResult(VNATimeResolvedDispersiveMeasurement1DResult):
    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._x_axis_units = "ns"

    def _model(self, t, A_r, A_i, T_2_ast, Delta_Omega, offset_r, offset_i, offset_r2, offset_i2, phase1, phase2):
        return -(A_r * np.cos(Delta_Omega * t + phase1) + 1j * A_i * np.cos(
            Delta_Omega * t + phase2)) * np.exp(-1 / T_2_ast * t) \
               + offset_r + 1j * offset_i + (1 - np.exp(-1 / T_2_ast * t)) * (
                           offset_r2 + 1j * offset_i2)

    def _generate_fit_arguments(self, x, data):
        amp_r, amp_i = np.ptp(np.real(data)) / 2, np.ptp(np.imag(data)) / 2
        if np.abs(np.max(np.real(data)) - np.real(data[0])) < np.abs(
                np.real(data[0]) - np.min(np.real(data))):
            amp_r = -amp_r
        if np.abs(np.max(np.imag(data)) - np.imag(data[0])) < np.abs(
                np.imag(data[0]) - np.min(np.imag(data))):
            amp_i = -amp_i
        offset_r, offset_i = np.max(np.real(data)) - np.abs(amp_r), \
                             np.max(np.imag(data)) - np.abs(amp_i)

        time_step = x[1] - x[0]
        max_frequency = 1 / time_step / 5
        min_frequency = 1e-4  # GHz
        frequency = float(np.random.random(1)) * (max_frequency -
                                                  min_frequency) + min_frequency
        T_2_ast = 1000

        m1p1 = np.array([-1, 1])
        p0_dict = OrderedDict(
            [
                ("amp_r", amp_r),
                ("amp_i", amp_i),
                ("T_2_ast", T_2_ast),
                ("Omega_R", frequency),
                ("offset_r", offset_r),
                ("offset_i", offset_i),
                ("offset_r2", offset_r),
                ("offset_i2", offset_i),
                ("phase1", 0),
                ("phase2", 0),
            ]
        )
        bounds_dict = OrderedDict(
            [
                ("amp_r", 1.5 * np.abs(amp_r) * m1p1),
                ("amp_i", 1.5 * np.abs(amp_i) * m1p1),
                ("T_2_ast", [5, 100e3]),
                ("Delta_Omega",
                 2 * np.pi * np.array([min_frequency, max_frequency])),
                ("offset_r", 1.5 * np.max(np.abs(np.real(data))) * m1p1),
                ("offset_i", 1.5 * np.max(np.abs(np.real(data))) * m1p1),
                ("offset_r2", 1.5 * np.max(np.abs(np.real(data))) * m1p1),
                ("offset_i2", 1.5 * np.abs(np.max(np.real(data))) * m1p1),
                ("phase1", 2 * np.pi * m1p1),
                ("phase2", 2 * np.pi * m1p1),
            ]
        )

        p0 = list(p0_dict.values())
        bounds = tuple(
            map(list, np.array(list(bounds_dict.values())).T)
        )

        return p0, bounds

    def get_T_2_ast(self):
        return self._fit_params[2], self._fit_errors[2]

    def get_basis(self):
        fit = self._fit_params
        A_r, A_i, offset_r, offset_i = fit[0], fit[1], fit[-2], fit[-1]
        ground_state = -A_r+offset_r+1j*(-A_i+offset_i)
        excited_state = A_r+offset_r+1j*(A_i+offset_i)
        return np.array((ground_state, excited_state))

    def get_ramsey_frequency(self):
        return self._fit_params[3]/2/np.pi*1e6, self._fit_errors[3]/2/np.pi*1e6

    def get_ramsey_decay(self):
        return self._fit_params[2], self._fit_errors[2]

    def _generate_annotation_string(self, opt_params, err):
        return "$T_2^*=%.2f \pm %.2f $ns\n$|\Delta\omega/2\pi| = %.3f \pm " \
               "%.3f$ MHz"%\
            (opt_params[2], err[2], 1e3*opt_params[3]/2/np.pi,
             1e3*err[3]/2/np.pi)


class DirectRamsey2D(DirectRamseyBase):
    """
        This class measures the oscillations after the driving pulse and
        sweeps either pulse's if_freq or the bias. Use it to measure
        two-dimensional plots with Ramsey chevrons.
    """

    def _init_measurement_result(self):
        self._measurement_result = DirectRamsey2D_Result(self._name,
                                                         self._sample_name)
        self._measurement_result._if_freq = \
            self._q_iqawg[0].get_calibration().get_if_frequency()

    def sweep_lo_shift(self, shifts, ro_cals, downconv_cals, pi_durations):
        """
        This method sets up the pulse's driving if_freq as the sweep
        parameter. Alternatively you can sweep the bias with
        sweep_current method.

        Parameters
        ----------
        shifts: np.array
            A list of frequncy shifts from the major LO if_freq.
        ro_cals: list
            A list of up-conversion IQ mixer's calibrations. The if_freq
            shift must be already included into every calibration.
        downconv_cals: list
            A list of down-conversion IQ mixer's calibrations. The if_freq
            shift and the delay through the fridge must be already included
            into every calibration.
        pi_durations: list
            A list of pi-pulse durations, which are used to calculate
            pi/2-pulse durations. The pi pulse duration depends on
            the driving if_freq. Nevertheless, you need not measure
            pi-pulse duration for every if_freq, because the oscillations
            are still visible and give a good result even for a non-perfect
            pi/2-pulse, although have a lower amplitude. Supplying a list of
            same pi-pulse durations will result in white spots on the plot.
        """
        self._name += "_lo"
        self._ro_cals = ro_cals
        self._downconv_cals = downconv_cals
        self._shifts = shifts
        self._pi_durations = pi_durations
        swept_pars = {"LO shift, Hz": (self._set_lo_shift, shifts)}
        self.set_swept_parameters(**swept_pars)
        self._setup_delay_parameter()

    def _setup_delay_parameter(self):
        delay_max = self._pulse_sequence_parameters["readout_duration"]
        delays = np.arange(
            0, delay_max,
            1 / self._dig[0].get_sample_rate() * 1e9
        )  # ns
        meas_data = self._measurement_result.get_data()
        # if_freq is already set in call of 'super()' class method
        meas_data["Delay, ns"] = delays
        self._measurement_result.set_data(meas_data)

    def sweep_current(self, currents):
        """
        This method sets up the bias which shifts the energy of the
        qubit. This method doesn't change any if_freq in the measurement
        equipment, easier to use and provides cleaner plots.
        Alternatively you can sweep the driving if_freq with sweep_current
        method.

        Parameters
        ----------
        currents: np.array
            A list of currents in Amper
        """
        self._name += '_loop'
        swept_pars = {"Current, A": (self._set_current, currents)}
        self.set_swept_parameters(**swept_pars)
        self._setup_delay_parameter()
        if self._src[0].set_status(1):
            print(f"Current source is on")

    def _set_lo_shift(self, shift):
        idx = np.abs(self._shifts - shift).argmin()
        ro_cal = self._ro_cals[idx]
        self._q_lo[0].set_frequency(ro_cal.get_lo_frequency())
        self._q_iqawg[0].set_parameters({"calibration": ro_cal})
        self._down_conversion_calibration = self._downconv_cals[idx]
        self._pulse_sequence_parameters["pi_half_pulse_duration"] = \
            self._pi_durations[idx] / 2
        self._output_pulse_sequence()

    def _set_current(self, current):
        self._src[0].set_current(current)
        self._output_pulse_sequence()

    def _recording_iteration(self):
        time, data = self._single_measurement()

        if self._pi_subtraction:
            time, data_pi = self._measure_pi_shifted()
            data = data + data_pi

        if_freq = self._q_iqawg[0].get_calibration().get_if_frequency()
        self._backup = data.copy()
        data = (data - np.mean(data)) * np.exp(-1j * 2 * np.pi * if_freq *
                                             time * 1e-9)

        if self._digital_filtering:
            b = signal.firwin(len(data), self._fir_cutoff,
                              fs=self._dig[0].get_sample_rate())
            data = signal.convolve(data, b, "same")

        data -= np.mean(data)

        return data


class DirectRamsey2D_Result(MeasurementResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._XX = None
        self._YY = None

        self._if_freq = None

    def set_parameter_name(self, parameter_name):
        self._parameter_name = parameter_name

    def _prepare_figure(self):
        fig = plt.figure(figsize=(17, 8))
        ax_map_re = plt.subplot2grid((1, 2), (0, 0))
        ax_map_im = plt.subplot2grid((1, 2), (0, 1))

        ax_map_re.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_map_re.set_ylabel("Ramsey delay, ns")
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
        return fig, (ax_map_re, ax_map_im), (cax_re, cax_im)

    def _plot(self, data):
        ax_map_re, ax_map_im = self._axes
        cax_re = self._caxes[0]
        cax_im = self._caxes[1]
        if "data" not in data.keys():
            return

        XX, YY, Z_re, Z_im = self._prepare_data_for_plot(data)

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

        im_map = ax_map_im.imshow(Z_im, origin='lower', cmap="RdBu_r",
                                     aspect='auto', vmax=im_max,
                                     vmin=-im_max, extent=extent)
        cax_im.cla()
        plt.colorbar(im_map, cax=cax_im)
        cax_im.tick_params(axis='y', right=False, left=True,
                           labelleft=True, labelright=False, labelsize='10')

    def _prepare_data_for_plot(self, data):
        complex_data = data["data"]
        re_data = np.real(complex_data).transpose()
        im_data = np.imag(complex_data).transpose()

        if self._XX is None and self._YY is None:
            self._YY = data[self._parameter_names[0]]
            self._XX = data["Delay, ns"]
        return self._XX, self._YY, re_data, im_data
