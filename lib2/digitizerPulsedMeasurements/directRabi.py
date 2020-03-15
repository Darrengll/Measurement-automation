import numpy as np
from typing import Union, List
from drivers.IQAWG import IQAWG
from drivers.Spectrum_m4x import SPCM
from drivers.E8257D import EXG, MXG
from lib2.digitizerPulsedMeasurements import digitizerTimeResolvedDirectMeasurement


# DEVELOPMENT BLOCK
from . import digitizerTimeResolvedDirectMeasurement
from importlib import reload
reload(digitizerTimeResolvedDirectMeasurement)

from .digitizerTimeResolvedDirectMeasurement import DigitizerTimeResolvedDirectMeasurement
from ..VNATimeResolvedDispersiveMeasurement1D import VNATimeResolvedDispersiveMeasurement1DResult
from ..IQPulseSequence import IQPulseBuilder


class DirectRabiBase(DigitizerTimeResolvedDirectMeasurement):
    def __init__(self, name, sample_name, plot_update_interval=1,
                 q_lo=[], q_iqawg=[], dig=[]):
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
                            "dig": dig}

        super().__init__(name, sample_name, devs_aliases_map, plot_update_interval)
        self._measurement_result = None  # has to be initialized in child classes
        # initialize 'self._measurement_result' that is specific for particular child class
        self._init_measurement_result()
        self._sequence_generator = IQPulseBuilder.build_direct_rabi_sequences

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
        longest_excitaion = np.max(self._swept_pars["excitation duration"][1])
        start_delay = self._pulse_sequence_parameters["start_delay"]
        return start_delay + longest_excitaion


class DirectRabiFromPulseDuration(DirectRabiBase):
    """
        Rabi from pulse duration measurements
    """
    def _init_measurement_result(self):
        self._measurement_result = RabiFromPulseDurationResult(self._name, self._sample_name)

    def sweep_excitation_durations(self, excitation_durations):
        super().set_swept_parameters(**{"excitation duration": (self._set_duration, excitation_durations)})

    def _set_duration(self, excitation):
        self._pulse_sequence_parameters["excitation_duration"] = excitation
        self._output_pulse_sequence()


class RabiFromPulseDurationResult(VNATimeResolvedDispersiveMeasurement1DResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._x_axis_units = "ns"

    def _model(self, t, A_r, A_i, T_R, Omega_R, offset_r, offset_i, offset_r2, offset_i2, phase1, phase2):
        return -(A_r * np.cos(Omega_R * t + phase1) + 1j * A_i * np.cos(Omega_R * t + phase2)) * np.exp(-1 / T_R * t)\
               + offset_r + 1j * offset_i + (1 - np.exp(-1 / T_R * t)) * (offset_r2 + 1j*offset_i2)

    def _generate_fit_arguments(self, x, data):
        amp_r, amp_i = np.ptp(np.real(data))/2, np.ptp(np.imag(data))/2
        if np.abs(np.max(np.real(data)) - np.real(data[0])) < np.abs(np.real(data[0]) - np.min(np.real(data))):
            amp_r = -amp_r
        if np.abs(np.max(np.imag(data)) - np.imag(data[0])) < np.abs(np.imag(data[0]) - np.min(np.imag(data))):
            amp_i = -amp_i
        offset_r, offset_i = np.max(np.real(data)) - np.abs(amp_r), np.max(np.imag(data)) - np.abs(amp_i)

        time_step = x[1] - x[0]
        max_frequency = 1 / time_step / 5
        min_frequency = 1e-4
        frequency = np.random.random(1) * (max_frequency - min_frequency) + min_frequency
        p0 = [amp_r, amp_i, 1000, frequency * 2 * np.pi, offset_r, offset_i, offset_r, offset_r, 0,0]

        bounds = ([-np.abs(amp_r) * 1.5, -np.abs(amp_i) * 1.5, 100,
                   min_frequency * 2 * np.pi, -10, -10, -10, -10, -np.pi, -np.pi],
                  [np.abs(amp_r) * 1.5, np.abs(amp_i) * 1.5, 100000,
                   max_frequency * 2 * np.pi, 10, 10, 10, 10, np.pi, np.pi])
        return p0, bounds

    def _generate_annotation_string(self, opt_params, err):
        return f"$T_R={opt_params[2]*1e-3:.2f}\pm {err[2]*1e-3:.2f}~\mu$s\n" \
               f"$\Omega_R/2\pi={opt_params[3] * 1e3 / 2 / np.pi:.2f}\pm {err[3] * 1e3 / 2 / np.pi:.2f}$ MHz\n" \
               f"$\Delta\phi={np.mod(opt_params[6] - opt_params[7], 2 * np.pi) - np.pi:.2f}$ rad"

    def _prepare_data_for_plot(self, data):
        return data[self._parameter_names[0]], data["data"]

    def get_pi_pulse_duration(self):
        return np.pi / self._fit_params[3] # ns

    def get_rabi_decay(self):
        return (self._fit_params[2] * 1e-3, self._fit_errors[2] * 1e-3)

    def get_rabi_frequency(self):
        return (self._fit_params[3] * 1e-3, self._fit_errors[3] * 1e-3)

    def get_basis(self):
        fit = self._fit_params
        A_r, A_i, offset_r, offset_i = fit[0], fit[1], fit[-4], fit[-3]
        ground_state = -A_r+offset_r+1j*(-A_i+offset_i)
        excited_state = A_r+offset_r+1j*(A_i+offset_i)
        return np.array((ground_state, excited_state))

    def get_betas(self):
        return [self._fit_params[0] + 1j*self._fit_params[1],  # beta_II
                self._fit_params[4] + 1j * self._fit_params[5]]  # beta_ZI or beta IZ depending on the qubit number in a qubit pair


class DirectRabiFromAmplitude(DirectRabiBase):
    """
    Rabi from pulse amplitude measurements
    """
    def _init_measurement_result(self):
        self._measurement_result = DirectRabiFromAmplitudeResult(self._name, self._sample_name)

    def sweep_amplitudes(self, amplitudes):
        super().set_swept_parameters(**{"amplitude": (self._set_amplitude, amplitudes)})

    def _set_amplitude(self, amplitude):
        self._pulse_sequence_parameters["excitation_amplitude"] = amplitude
        self._output_pulse_sequence()


class DirectRabiFromAmplitudeResult(VNATimeResolvedDispersiveMeasurement1DResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._x_axis_units = "ratio"

    def _model(self, amplitude, A_r, A_i, pi_amplitude, offset_r, offset_i, phase_r, phase_i):
        return -(A_r * np.cos(np.pi * amplitude / pi_amplitude + phase_r)
                 + 1j * A_i * np.cos(np.pi * amplitude / pi_amplitude + phase_i)) + (offset_r + 1j * offset_i)

    def _generate_fit_arguments(self, x, data):
        amp_r, amp_i = np.ptp(np.real(data)) / 2, np.ptp(np.imag(data)) / 2
        if np.abs(np.max(np.real(data)) - np.real(data[0])) < np.abs(np.real(data[0]) - np.min(np.real(data))):
            amp_r = -amp_r
        if np.abs(np.max(np.imag(data)) - np.imag(data[0])) < np.abs(np.imag(data[0]) - np.min(np.imag(data))):
            amp_i = -amp_i
        offset_r, offset_i = np.max(np.real(data)) - np.abs(amp_r), np.max(np.imag(data)) - np.abs(amp_i)
        amp_step = x[1] - x[0]
        min_pi_pulse_amp = amp_step * 2 * 5
        max_pi_pulse_amp = (x[-1] - x[0]) * 2 * 10
        pi_pulse_amp = np.random.random(1) * (max_pi_pulse_amp - min_pi_pulse_amp) + min_pi_pulse_amp
        bounds = ([-np.abs(amp_r)*1.5,  -np.abs(amp_i)*1.5, min_pi_pulse_amp,   -10, -10, -np.pi, -np.pi],
                  [np.abs(amp_r)*1.5,   np.abs(amp_i)*1.5,  max_pi_pulse_amp,   10, 10, np.pi, np.pi])
        p0 = [amp_r, amp_i, pi_pulse_amp, offset_r, offset_i, 0, 0]
        return p0, bounds
    #
    # def _prepare_data_for_plot(self, data):
    #     return data["amplitude multiplier"], data["data"]

    def _generate_annotation_string(self, opt_params, err):
        return f"$(\pi) = {opt_params[2]:.2f} \pm {err[2]:.2f}$ a.u.\n" \
               f"$\Delta\phi={np.mod(opt_params[5] - opt_params[6], 2 * np.pi) - np.pi:.2f}$ rad"

    def get_pi_pulse_amplitude(self):
        return self._fit_params[2]

    def _prepare_data_for_plot(self, data):
        return data[self._parameter_names[0]], data["data"]

    def get_basis(self):
        fit = self._fit_params
        A_r, A_i, offset_r, offset_i = fit[0], fit[1], fit[-4], fit[-3]
        ground_state = -A_r + offset_r + 1j * (-A_i + offset_i)
        excited_state = A_r + offset_r + 1j * (A_i + offset_i)
        return np.array((ground_state, excited_state))