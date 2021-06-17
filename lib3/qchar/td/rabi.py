from collections import OrderedDict

import numpy as np

from drivers.Yokogawa_GS210 import Yokogawa_GS210
from drivers.IQAWG import AWGChannel
from lib2.IQPulseSequence import IQPulseBuilder
from lib2.VNATimeResolvedDispersiveMeasurement1D import \
    VNATimeResolvedDispersiveMeasurement1DResult
from lib3.qchar.td.digitizerTimeResolvedMeasurement import \
    DigitizerTimeResolvedMeasurement, FLUX_CONTROL_TYPE


class RabiBase(DigitizerTimeResolvedMeasurement):
    def __init__(self, name, sample_name, plot_update_interval=1,
                 q_lo=None, q_iqawg=None, ro_lo=None, ro_iqawg=None,
                 dig=None, src=None, vna=None,
                 save_traces=False,
                 flux_control_type=FLUX_CONTROL_TYPE.CURRENT):
        """

        Parameters
        ----------
        name : str
        sample_name : str
        plot_update_interval : float

        q_lo : List[EXG]
        q_iqawg : list[IQAWG]
        ro_lo : List[MXG]
        ro_iqawg : list[IQAWG]
        dig : list[SPCM]
        src : List[Union[AWGChannel, Yokogawa_GS210]]
        save_traces : bool
            Whether or not to keep each digitizer trace memory and saving
            process
        """
        devs_aliases_map = {"q_lo": q_lo,
                            "q_iqawg": q_iqawg,
                            "ro_lo": ro_lo,
                            "ro_iqawg": ro_iqawg,
                            "dig": dig,
                            "src": src,
                            "vna": vna}
        super().__init__(name, sample_name, devs_aliases_map,
                         plot_update_interval, save_traces,
                         flux_control_type=flux_control_type)
        # has to be initialized in child classes
        self._measurement_result = None
        # initialize 'self._measurement_result' that is
        # specific for particular child class
        self._init_measurement_result()
        self._sequence_generator = None  # initialize in child classes

    def _init_measurement_result(self):
        """
        Pure virtual function that allows child classes to initialize
        measurement_result attribute in a 'hook' fasion

        Returns
        -------
        None
        """
        raise NotImplementedError

    def _get_longest_pulse_sequence_duration(
            self, pulse_sequence_parameters, swept_pars
    ):
        """
        Implementation of purely virtual function for 'DirectRabi' sequences
        measurements. Function calculates and return the longest pulse
        sequence duration based on pulse sequence parameters provided and
        'self._sequence_generator' implementation.

        Parameters
        ----------
        pulse_sequence_parameters : dict
            Dictionary that contain pulse sequence parameters for which
            you wish to calculate the longest duration.
            This parameters are fixed.

        swept_pars : dict
            Sweep parameters that are needed for calculation of the
            longest sequence.

        Returns
        -------
        float
            Longest sequence duration based on pulse sequence parameters in ns.

        Notes
        ------
           This function is introduced in the context of the
        solution to the phase jumps, causedby clock incompatibility between
        AWG and digitizer.
            The aim is to fix distance between digitizer
        measurement window and AWG trigger that obtains digitizer. The last
        pulse ending should stay at fixed distance from trigger event in
        contrary with previous implementation, where the start of the first
        control pulse was fixed relative to trigger event.
           The previous solution forced digitizer acquisition
        window (which is placed after the pulse sequence, usually) to
        shift further in timeline following the extension of the length of the
        pulse sequence.
            And due to the fact that extension length does not always coincide
        with acquisition window displacement (due to difference in AWG and
        digitizer clock period) the phase jumps arise as a problem. The
        solution is that end of the last pulse stays at the same distance from
        the trigger event and pulse sequence
        length grows "back in timeline".
            Together with requirement that 'repetition_period" is dividable by
        both AWG and digitizer clocks this will ensure that phase jumps will be
        neglected completely.
        """
        longest_excitaion = None
        if "excitation_durations" in self._swept_pars:
            longest_excitaion = np.max(self._swept_pars[
                                           "excitation_durations"][1])
        elif "excitation_duration" in self._pulse_sequence_parameters:
            longest_excitaion = \
                self._pulse_sequence_parameters["excitation_duration"]
        else:
            raise ValueError("Cannot estimate longest pulse duration based on"
                             " 'self.pulse_sequence_parameters'"
                             " or 'self._swept_pars'"
                             "\n No pulse duration data found.")
        return longest_excitaion


class DispersiveRabiFromDuration(RabiBase):
    def __init__(self, name, sample_name, plot_update_interval=1,
                 q_lo=None, q_iqawg=None, ro_lo=None, ro_iqawg=None, dig=None,
                 src=None, vna=None,
                 save_traces=False,
                 flux_control_type=FLUX_CONTROL_TYPE.CURRENT):
        super().__init__(
            name, sample_name, plot_update_interval=plot_update_interval,
            q_lo=q_lo, q_iqawg=q_iqawg, ro_lo=ro_lo, ro_iqawg=ro_iqawg,
            dig=dig, src=src, vna=vna,
            save_traces=save_traces, flux_control_type=flux_control_type,
        )
        self._sequence_generator = \
            IQPulseBuilder.build_dispersive_rabi_sequences2
        self.__ctr = 0

    def _init_measurement_result(self):
        self.set_measurement_result(
            RabiFromPulseDurationResult(self._name, self._sample_name)
        )

    def sweep_durations(self, durations):
        self.set_swept_parameters(
            **{
                "excitation_durations": (self._set_duration, durations)
            }
        )

    def _set_duration(self, duration):
        self._pulse_sequence_parameters["excitation_duration"] = duration
        self._output_pulse_sequence()
        # self.__ctr += 1
        # if self.__ctr % 20 == 0:
        #     self._q_iqawg[0]._channels[0]._host_awg.plot_waveforms()


class RabiFromPulseDurationResult(
    VNATimeResolvedDispersiveMeasurement1DResult
):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._x_axis_units = "ns"

    def _model(self, t, a_r, a_i, b_r, b_i, t_1, t_2, omega_r, offset_r,
               offset_i, offset_r2, offset_i2):
        c1 = offset_r + 1j * offset_i
        c2 = a_r + 1j * a_i
        c3 = b_r + 1j * b_i
        c4 = offset_r2 + 1j * offset_i2
        return c1 + (c2 * np.cos(omega_r * t) + c3 / omega_r *
                     np.sin(omega_r * t)) * np.exp(-t * (1 / t_1 + 1 / t_2)) \
               + c4 * np.exp(-t / t_2)

    # can be customized for different models
    def _generate_fit_arguments(self, x, data):
        amp_r, amp_i = np.ptp(np.real(data)) / 2, np.ptp(
            np.imag(data)) / 2  # > 0
        offset_r, offset_i = np.mean(np.real(data)), np.mean(np.imag(data))

        time_step = x[1] - x[0]
        max_frequency = 1 / time_step / 5  # now less than 5 points per period
        min_frequency = 1e-4  # GHz, = 0.1 MHz
        # np.random.random returns scalar
        frequency = np.random.random(1)[0] * (max_frequency -
                                              min_frequency) + min_frequency
        t_0 = 100

        p0_dict = OrderedDict(
            [
                ("A_r", amp_r),
                ("A_i", amp_i),
                ("B_r", amp_r * frequency),
                ("B_i", amp_i * frequency),
                ("T1", t_0),
                ("T2", 2 * t_0),
                ("Omega_R", 2 * np.pi * frequency),
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
        t_1 = opt_params[4]
        t_2 = opt_params[5]
        t_r = 2 * t_1 * t_2 / (t_1 + t_2)
        t_r_err = t_r ** 2 * (err[4] / t_1 ** 2 + err[5] / t_2 ** 2)
        om_r = opt_params[6] * 1e3 / 2 / np.pi
        om_r_err = err[6] * 1e3 / 2 / np.pi
        return (f"$T_R={t_1 * t_2 / (t_1 + t_2):.3f} \pm {t_r_err:.3f}$ ns\n"
                f"$\Omega_R/(2\pi)={om_r:.2f} \pm {om_r_err:.2f}$ MHz\n"
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
        return self._fit_params[2], self._fit_errors[2]

    def get_rabi_frequency(self):
        """
        Returns Omega_R and it's standard deviation from fit results in 'MHz'

        Returns
        -------
        tuple[float,float]
            tuple( Omega_R, standard deviation(Omega_R) )
        """
        return self._fit_params[3] * 1e3, self._fit_errors[3] * 1e3

    def get_phase_real(self):
        # return phase `phi` of the real part of oscillations `Cos(w t + phi)`
        a_r = self._fit_params[0]
        a_i = self._fit_params[1]
        b_r = self._fit_params[2]
        b_i = self._fit_params[3]
        omega = self._fit_params[6]
        a = a_r + 1j * a_i
        b = (b_r + 1j * b_i) / omega

        return np.arctan2(-np.real(b), np.real(a))

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
        ground_state = -A_r + offset_r + 1j * (-A_i + offset_i)
        excited_state = A_r + offset_r + 1j * (A_i + offset_i)
        return np.array((ground_state, excited_state))

    def get_betas(self):
        # beta_ZI or beta IZ depending on the qubit number in a qubit pair
        return [self._fit_params[0] + 1j * self._fit_params[1],  # beta_II
                self._fit_params[4] + 1j * self._fit_params[
                    5]]