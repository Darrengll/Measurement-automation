import numpy as np
from typing import Union, List
from drivers.IQAWG import IQAWG
from drivers.Spectrum_m4x import SPCM
from drivers.E8257D import EXG, MXG

from collections import OrderedDict

# DEVELOPMENT RELOAD BLOCK START #
from importlib import reload

from . import digitizerTimeResolvedDirectMeasurement
reload(digitizerTimeResolvedDirectMeasurement)
from .digitizerTimeResolvedDirectMeasurement import DigitizerTimeResolvedDirectMeasurement

from .. import VNATimeResolvedDispersiveMeasurement1D
reload(VNATimeResolvedDispersiveMeasurement1D)
from ..VNATimeResolvedDispersiveMeasurement1D import VNATimeResolvedDispersiveMeasurement1DResult

from .. import IQPulseSequence
reload(IQPulseSequence)
from ..IQPulseSequence import IQPulseBuilder

# DEVELOPMENT RELOAD BLOCK END #


class DirectRabiBase(DigitizerTimeResolvedDirectMeasurement):
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
                            "src": src}

        super().__init__(name, sample_name, devs_aliases_map,
                         plot_update_interval, save_traces)
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
        longest_excitaion = None
        if "excitation_durations" in self._swept_pars:
            longest_excitaion = np.max(self._swept_pars[
                                           "excitation_durations"][1])
        elif "excitation_duration" in self._pulse_sequence_parameters:
            longest_excitaion = self._pulse_sequence_parameters["excitation_duration"]
        else:
            raise ValueError("Cannot estimate longest pulse duration based on 'self.pulse_sequence_parameters'"
                             " or 'self._swept_pars'\n No pulse duration data found.")
        return longest_excitaion


class DirectRabiFromPulseDuration(DirectRabiBase):
    """
        Rabi from pulse duration measurements
    """
    def _init_measurement_result(self):
        self._measurement_result = RabiFromPulseDurationResult(self._name, self._sample_name)

    def sweep_excitation_durations(self, excitation_durations):
        super().set_swept_parameters(**{"excitation_durations": (
            self._set_duration, excitation_durations)})

    def _set_duration(self, excitation):
        self._pulse_sequence_parameters["excitation_duration"] = excitation
        self._output_pulse_sequence()


class RabiFromPulseDurationResult(VNATimeResolvedDispersiveMeasurement1DResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._x_axis_units = "ns"

    def _model(self, t, A_r, A_i, B_r, B_i, T1, T2, Omega_R, offset_r,
               offset_i, offset_r2, offset_i2):
        c1 = offset_r + 1j * offset_i
        c2 = A_r + 1j * A_i
        c3 = B_r + 1j * B_i
        c4 = offset_r2 + 1j * offset_i2
        return c1 + (c2 * np.cos(Omega_R * t) + c3 / Omega_R *
                     np.sin(Omega_R * t)) * np.exp(-t * (1 / T1 + 1 / T2)) + \
               c4 * np.exp(-t / T2)

    # can be customized for different models
    def _generate_fit_arguments(self, x, data):
        amp_r, amp_i = np.ptp(np.real(data))/2, np.ptp(np.imag(data))/2  # > 0
        offset_r, offset_i = np.mean(np.real(data)), np.mean(np.imag(data))

        time_step = x[1] - x[0]
        max_frequency = 1 / time_step / 5  # now less than 5 points per period
        min_frequency = 1e-4  # GHz, = 0.1 MHz
        # np.random.random returns scalar
        frequency = np.random.random(1)[0] * (max_frequency -
                                               min_frequency) + min_frequency
        T_0 = 100

        p0_dict = OrderedDict(
            [
                ("A_r", amp_r),
                ("A_i", amp_i),
                ("B_r", amp_r * frequency),
                ("B_i", amp_i * frequency),
                ("T1", T_0),
                ("T2", 2 * T_0),
                ("Omega_R", 2*np.pi*frequency),
                ("offset_r", offset_r),
                ("offset_i", offset_i),
                ("offset_r2", offset_r),
                ("offset_i2", offset_i),
            ]
        )

        m1p1 = np.array([-1, 1])
        bounds_dict = OrderedDict(
            [
                ("A_r", 1.5 * np.abs(amp_r) * m1p1),
                ("A_i", 1.5 * np.abs(amp_i) * m1p1),
                ("B_r", 1.5 * np.abs(amp_r) * max_frequency * m1p1),
                ("B_i", 1.5 * np.abs(amp_i) * max_frequency * m1p1),
                ("T1", [5, 100e3]),
                ("T2", [5, 100e3]),
                ("Omega_R",
                 2 * np.pi * np.array([min_frequency, max_frequency])),
                ("offset_r", 1.1 * np.max(np.abs(np.real(data))) * m1p1),
                ("offset_i", 1.1 * np.max(np.abs(np.imag(data))) * m1p1),
                ("offset_r2", 2 * np.max(np.abs(np.real(data))) * m1p1),
                ("offset_i2", 2 * np.max(np.abs(np.imag(data))) * m1p1),
            ]
        )

        p0 = list(p0_dict.values())
        bounds = tuple(
            map(list, np.array(list(bounds_dict.values())).T)
        )

        return p0, bounds

    def _generate_annotation_string(self, opt_params, err):
        T1 = opt_params[4]
        T2 = opt_params[5]
        T_R = 2 * T1 * T2 / (T1 + T2)
        T_R_err = T_R**2 * (err[4] / T1**2 + err[5] / T2**2)
        Om_R = opt_params[6] * 1e3 / 2 / np.pi
        Om_R_err = err[6] * 1e3 / 2 / np.pi
        return (f"$T_R={T1 * T2/(T1 + T2):.3f} \pm {T_R_err:.3f}~\mu$s\n"
                f"$\Omega_R/(2\pi)={Om_R:.2f} \pm {Om_R_err:.2f}$ MHz\n"
                rf"$\Delta \varphi$ = {self.get_phase_diff():.2f} rad")

    def _prepare_data_for_plot(self, data):
        return data[self._parameter_names[0]], data["data"]

    def get_pi_pulse_duration(self):
        return np.pi / self._fit_params[6]  # ns

    def get_rabi_decay(self):
        """
        Returns T_R and it's standard deviation from fit results in 'ns'.

        Returns
        -------
        tuple[float,float]
            tuple( T_R, standard deviation(T_R) )
        """
        return (self._fit_params[2], self._fit_errors[2])

    def get_rabi_frequency(self):
        """
        Returns Omega_R and it's standard deviation from fit results in 'MHz'

        Returns
        -------
        tuple[float,float]
            tuple( Omega_R, standard deviation(Omega_R) )
        """
        return (self._fit_params[3] * 1e3, self._fit_errors[3] * 1e3)

    def get_phase_real(self):
        # return phase `phi` of the real part of oscillations `Cos(w t + phi)`
        A_r = self._fit_params[0]
        A_i = self._fit_params[1]
        B_r = self._fit_params[2]
        B_i = self._fit_params[3]
        Omega = self._fit_params[6]
        A = A_r + 1j * A_i
        B = (B_r + 1j * B_i)/Omega

        return np.arctan2(-np.real(B), np.real(A))

    def get_phase_imag(self):
        # return phase `phi` of the imaginary part of oscillations
        # `Cos(w t + phi)`
        A_r = self._fit_params[0]
        A_i = self._fit_params[1]
        B_r = self._fit_params[2]
        B_i = self._fit_params[3]
        Omega = self._fit_params[6]
        A = A_r + 1j * A_i
        B = (B_r + 1j * B_i) / Omega

        return np.arctan2(-np.imag(B), np.imag(A))

    def get_phase_diff(self):
        # TODO: not working properly (see visuzalization and compare with this)
        # return relative phase difference between cosine fits of
        # real and imaginary parts
        return self.get_phase_real() - self.get_phase_imag()

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

    def _model(self, amplitude, A_r, A_i, pi_amplitude, offset_r, offset_i, phase_r, phase_i,
               offset_r1, offset_i1, amp_offset_decay_rate):
        return -(A_r * np.cos(np.pi * amplitude / pi_amplitude + phase_r) +\
                 + 1j * A_i * np.cos(np.pi * amplitude / pi_amplitude + phase_i)) + (offset_r + 1j * offset_i) + \
               (offset_r1 + 1j*offset_i1)*(1 - np.exp(amp_offset_decay_rate * amplitude))

    def _generate_fit_arguments(self, x, data):
        amp_r, amp_i = np.ptp(np.real(data)) / 2, np.ptp(np.imag(data)) / 2
        offset_r, offset_i = np.mean(np.real(data)), np.mean(np.real(data))
        amp_step = x[1] - x[0]
        min_pi_pulse_amp = amp_step * 2 * 5
        max_pi_pulse_amp = (x[-1] - x[0]) * 2 * 10
        pi_pulse_amp = np.random.random(1) * (max_pi_pulse_amp - min_pi_pulse_amp) + min_pi_pulse_amp
        bounds = ([-np.abs(amp_r)*1.5,  -np.abs(amp_i)*1.5, min_pi_pulse_amp,   -10, -10, -np.pi, -np.pi, -10, -10, 0],
                  [np.abs(amp_r)*1.5,   np.abs(amp_i)*1.5,  max_pi_pulse_amp,   10, 10, np.pi, np.pi, 10, 10, 1])
        p0 = [amp_r, amp_i, pi_pulse_amp, offset_r, offset_i, 0, 0, offset_r, offset_i, 0.001]
        return p0, bounds
    #
    # def _prepare_data_for_plot(self, data):
    #     return data["amplitude multiplier"], data["data"]

    def _generate_annotation_string(self, opt_params, err):
        return f"$(\pi) = {opt_params[2]:.3f} \pm {err[2]:.3f}$ a.u.\n" \
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